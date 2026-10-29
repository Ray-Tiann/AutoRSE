from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import const, utils

from .BaseModel import BaseModel
from .intsr_layers import DSFNet, QueryDrivenEncoder, build_intsr_masks


class IntSR(BaseModel):
    """Public-data reimplementation of IntSR on the UniSAR data pipeline.

    The released UniSAR artifacts contain tokenized queries but not the raw
    query text or the frozen Qwen3-0.6B representations used by IntSR.  This
    supports token queries and frozen features keyed by public session IDs.
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--intsr_dim', type=int, default=32)
        parser.add_argument('--qdb_layers', type=int, default=3)
        parser.add_argument('--max_relative_position', type=int, default=256)
        parser.add_argument('--relative_time_buckets', type=int, default=32)
        parser.add_argument('--absolute_time_buckets', type=int, default=4096)
        parser.add_argument('--dsf_num_scenarios', type=int, default=2)
        parser.add_argument('--dsf_layers', type=int, default=3)
        parser.add_argument(
            '--prediction_mode',
            choices=('ranking_mlp', 'retrieval_dot'),
            default='ranking_mlp',
            help='Candidate-conditioned ranking logits (Appendix C) or '
                 'retrieval InfoNCE scores from Eq. (14).',
        )
        parser.add_argument(
            '--historical_query_include_item', type=int, choices=(0, 1),
            default=1,
            help='Whether historical Q tokens include their interacted item.',
        )
        parser.add_argument(
            '--candidate_scorer_layers', type=int, choices=(1, 2), default=1,
            help='Number of layers in the candidate-ranking MLP.',
        )
        parser.add_argument(
            '--separate_task_embedding', type=int, choices=(0, 1), default=0,
            help='Use a dedicated S&R task-tag embedding in DSF scenario '
                 'features instead of sharing the feedback-token table.',
        )
        parser.add_argument('--query_backend',
                            choices=('token', 'query_id', 'precomputed_qwen'),
                            default='token')
        parser.add_argument('--query_feature_path', type=str, default='',
                            help='Frozen .npy session features or compact '
                                 '.npz query features plus session index.')
        parser.add_argument(
            '--session_pos_items_path', type=str, default='',
            help='Explicit .npy [session, positive_item] table used only by '
                 'precomputed_qwen to avoid loading the full session vocabulary.')
        parser.add_argument(
            '--normalize_precomputed_queries',
            type=int,
            choices=(0, 1),
            default=1,
            help='L2-normalize frozen LLM features before the learned '
                 'projection. This keeps their scale comparable to item '
                 'and token embeddings.',
        )
        parser.add_argument(
            '--intsr_embedding_init_std',
            type=float,
            default=0.02,
            help='Initialization std for IntSR token, feedback, and time '
                 'embeddings.',
        )
        parser.add_argument(
            '--query_augmentation_beta',
            type=float,
            default=0.0,
            help='Probability of replacing a historical non-search Q with '
                 'an item-associated reconstructed query during training.',
        )
        parser.add_argument(
            '--query_augmentation_pool_path',
            type=str,
            default='',
            help='Train-only .npz item-to-query pool. Required when query '
                 'augmentation is enabled.',
        )
        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.hidden_dim = args.intsr_dim
        self.prediction_mode = getattr(args, 'prediction_mode', 'ranking_mlp')
        self.historical_query_include_item = bool(
            getattr(args, 'historical_query_include_item', 1))
        self.candidate_scorer_layers = getattr(
            args, 'candidate_scorer_layers', 1)
        self.separate_task_embedding = bool(
            getattr(args, 'separate_task_embedding', 0))
        self.query_backend = args.query_backend
        self.normalize_precomputed_queries = bool(
            getattr(args, 'normalize_precomputed_queries', 1))
        self.embedding_init_std = getattr(
            args, 'intsr_embedding_init_std', 0.02)
        if self.embedding_init_std <= 0:
            raise ValueError('intsr_embedding_init_std must be positive.')
        self.query_augmentation_beta = getattr(
            args, 'query_augmentation_beta', 0.0)
        if not 0.0 <= self.query_augmentation_beta <= 1.0:
            raise ValueError('query_augmentation_beta must be in [0, 1].')
        self._query_augmentation_generator = None
        if self.query_augmentation_beta > 0:
            # Keep augmentation sampling off PyTorch's global RNG stream so
            # beta ablations do not silently change every later dropout mask.
            self._query_augmentation_generator = torch.Generator(
                device=self.device)
            self._query_augmentation_generator.manual_seed(
                int(getattr(args, 'random_seed', 0)) + 104729)
        self._precomputed_query_features = None
        self._precomputed_query_index = None
        self._precomputed_query_archive = None
        self.register_buffer(
            'item_query_feature_ids', None, persistent=False)
        self.register_buffer(
            'item_query_feature_counts', None, persistent=False)
        self.register_buffer(
            'session_query_ids', None, persistent=False)

        if self.query_backend == 'token':
            if self.query_augmentation_beta > 0:
                raise ValueError(
                    'Query augmentation requires precomputed_qwen.')
            query_input_dim = self.query_size
        elif self.query_backend == 'query_id':
            if self.query_augmentation_beta > 0:
                raise ValueError(
                    'Query augmentation requires precomputed_qwen.')
            query_count = self._build_session_query_ids()
            self.query_id_embedding = nn.Embedding(
                query_count, self.query_size, padding_idx=0)
            # The inherited word encoder is unused by this backend (and Amazon
            # item features do not share it), so keep it out of the optimizer.
            for parameter in self.query_embedding.parameters():
                parameter.requires_grad = False
            query_input_dim = self.query_size
        else:
            query_input_dim = self._load_precomputed_query_features(
                args.query_feature_path)
            if self.query_augmentation_beta > 0:
                self._load_item_query_pool(
                    getattr(args, 'query_augmentation_pool_path', ''))

        self.item_projection = nn.Linear(self.item_size, self.hidden_dim)
        self.query_projection = nn.Linear(query_input_dim, self.hidden_dim)
        self.user_projection = nn.Linear(self.user_size, self.hidden_dim)

        self.universal_query = nn.Parameter(torch.empty(self.hidden_dim))
        self.feedback_embedding = nn.Embedding(3, self.hidden_dim,
                                               padding_idx=0)
        self.token_type_embedding = nn.Embedding(4, self.hidden_dim)
        self.absolute_time_embedding = nn.Embedding(
            args.absolute_time_buckets, self.hidden_dim)
        self.absolute_time_buckets = args.absolute_time_buckets

        max_events = getattr(const, 'max_all_his_len',
                             const.max_rec_his_len + const.max_src_session_his_len)
        max_position = max(args.max_relative_position, max_events * 4 + 2)
        self.qdb = QueryDrivenEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=args.qdb_layers,
            dropout=self.dropout,
            max_position=max_position,
            time_buckets=args.relative_time_buckets,
        )

        scenario_dim = self.hidden_dim * 3
        self.dsf = DSFNet(
            hidden_dim=self.hidden_dim,
            scenario_dim=scenario_dim,
            num_scenarios=args.dsf_num_scenarios,
            num_layers=args.dsf_layers,
            dropout=self.dropout,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        if self.candidate_scorer_layers == 1:
            self.candidate_scorer = nn.Linear(self.hidden_dim, 1)
        else:
            # Appendix C maps each candidate-conditioned QDB output to a
            # ranking logit with an MLP. Keep the linear head available for
            # controlled comparisons with earlier checkpoints.
            self.candidate_scorer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        if self.prediction_mode == 'retrieval_dot':
            for parameter in self.candidate_scorer.parameters():
                parameter.requires_grad = False
        self._init_weights()
        if self.separate_task_embedding:
            # Preserve every existing parameter's initialization so this flag
            # remains a strict one-variable ablation.
            rng_state = torch.random.get_rng_state()
            self.task_embedding = nn.Embedding(
                3, self.hidden_dim, padding_idx=0)
            torch.random.set_rng_state(rng_state)
        self._init_intsr_embeddings()
        self.to(self.device)

    def _init_intsr_embeddings(self):
        """Initialize IntSR-only embeddings on the same compact scale.

        ``BaseModel._init_weights`` intentionally skips ``nn.Embedding``
        modules because the inherited UniSAR feature encoders initialize
        their own tables. The IntSR-specific tables therefore otherwise keep
        PyTorch's unit-variance default, which overwhelms projected item
        embeddings and makes candidate-conditioned ranking nearly constant.
        """
        nn.init.normal_(self.universal_query, std=self.embedding_init_std)
        for embedding in (
                self.feedback_embedding,
                self.token_type_embedding,
                self.absolute_time_embedding):
            nn.init.normal_(embedding.weight, std=self.embedding_init_std)
        if hasattr(self, 'task_embedding'):
            rng_state = torch.random.get_rng_state()
            nn.init.normal_(self.task_embedding.weight,
                            std=self.embedding_init_std)
            torch.random.set_rng_state(rng_state)
        if hasattr(self, 'query_id_embedding'):
            nn.init.normal_(self.query_id_embedding.weight,
                            std=self.embedding_init_std)
        with torch.no_grad():
            self.feedback_embedding.weight[0].zero_()
            if hasattr(self, 'task_embedding'):
                self.task_embedding.weight[0].zero_()
            if hasattr(self, 'query_id_embedding'):
                self.query_id_embedding.weight[0].zero_()

    def _build_session_query_ids(self) -> int:
        """Map every search session to its deduplicated public query ID.

        ``query_vocab.pkl`` is the public query-ID table: Amazon has hundreds
        of thousands of search sessions but only a few thousand distinct token
        sequences. The released session vocabulary stores the token sequence
        per session rather than the corresponding table row, so recover that
        row with a vectorized, verified 64-bit hash join. No labels or target
        items participate in this mapping.
        """
        query_vocab = utils.load_pickle(const.query_vocab)
        session_vocab = utils.load_pickle(const.session_map_vocab)
        session_keywords = np.asarray(session_vocab['keyword'], dtype=np.int64)
        if session_keywords.ndim != 2:
            raise ValueError('Session keyword table must be two-dimensional.')

        padded_queries = np.zeros(
            (len(query_vocab), session_keywords.shape[1]), dtype=np.int64)
        for query_id, tokens in enumerate(query_vocab):
            tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
            length = min(len(tokens), session_keywords.shape[1])
            padded_queries[query_id, :length] = tokens[:length]
        if len(padded_queries) == 0 or np.any(padded_queries[0]):
            raise ValueError('Query ID zero must be the all-zero padding query.')

        mask64 = (1 << 64) - 1
        coefficients = np.asarray([
            0x9E3779B185EBCA87
            ^ ((position * 0xC2B2AE3D27D4EB4F) & mask64)
            for position in range(session_keywords.shape[1])
        ], dtype=np.uint64)
        query_hashes = (
            padded_queries.astype(np.uint64) * coefficients
        ).sum(axis=1, dtype=np.uint64)
        if len(np.unique(query_hashes)) != len(query_hashes):
            raise ValueError('Query hash collision in public query vocabulary.')

        session_hashes = np.empty(len(session_keywords), dtype=np.uint64)
        chunk_size = 65536
        for start in range(0, len(session_keywords), chunk_size):
            chunk = session_keywords[start:start + chunk_size]
            session_hashes[start:start + len(chunk)] = (
                chunk.astype(np.uint64) * coefficients
            ).sum(axis=1, dtype=np.uint64)
        order = np.argsort(query_hashes)
        positions = np.searchsorted(query_hashes[order], session_hashes)
        if np.any(positions >= len(order)):
            raise ValueError('Session query is absent from query_vocab.pkl.')
        session_query_ids = order[positions]
        if not np.array_equal(
                padded_queries[session_query_ids], session_keywords):
            raise ValueError('Session query is absent from query_vocab.pkl.')
        if session_query_ids[0] != 0:
            raise ValueError('Padding session must map to query ID zero.')

        self.session_query_ids = torch.from_numpy(
            session_query_ids.astype(np.int64, copy=False))
        return len(query_vocab)
    def _load_precomputed_query_features(self, feature_path: str) -> int:
        if not feature_path:
            raise ValueError(
                'query_feature_path is required for precomputed_qwen.')
        path = Path(feature_path)
        if path.suffix not in ('.npy', '.npz'):
            raise ValueError(
                'Precomputed query features must be .npy or .npz.')
        if not path.is_file():
            raise FileNotFoundError(
                'Precomputed query feature file not found: {}'.format(path))

        expected_rows = self.session_map_vocab['pos_items'].size(0)
        if path.suffix == '.npz':
            archive = np.load(path, allow_pickle=False)
            required = {'features', 'session_to_feature'}
            if not required.issubset(archive.files):
                raise ValueError(
                    'Compact query archive requires features and '
                    'session_to_feature arrays.')
            features = archive['features']
            index = archive['session_to_feature']
            if index.ndim != 1 or len(index) != expected_rows:
                raise ValueError(
                    'session_to_feature must have one row per session.')
            if not np.issubdtype(index.dtype, np.integer):
                raise ValueError('session_to_feature must use integer IDs.')
            if index.size and (
                    index.min() < 0 or index.max() >= len(features)):
                raise ValueError(
                    'session_to_feature contains an invalid feature row.')
            if index[0] != 0:
                raise ValueError(
                    'session_to_feature row 0 must map to padding row 0.')
            self._precomputed_query_archive = archive
            self._precomputed_query_index = index
        else:
            features = np.load(path, mmap_mode='r')
            if features.shape[0] != expected_rows:
                raise ValueError(
                    'Query feature row count {} does not match session '
                    'vocabulary size {}.'.format(
                        features.shape[0], expected_rows))
        if features.ndim != 2:
            raise ValueError(
                'Query feature matrix must have shape [rows, dimension].')
        if features.shape[1] <= 0:
            raise ValueError('Query feature dimension must be positive.')
        if not np.issubdtype(features.dtype, np.floating):
            raise ValueError('Query features must use a floating-point dtype.')
        if not np.isfinite(features[0]).all() or not np.allclose(
                features[0], 0):
            raise ValueError(
                'Query feature row 0 must be finite all-zero padding.')

        # Keep the frozen table as a read-only CPU mmap. It is deliberately
        # excluded from module buffers and model checkpoints.
        self._precomputed_query_features = features
        return int(features.shape[1])

    def _load_item_query_pool(self, pool_path: str):
        """Load an item-to-query pool constructed only from training rows."""
        if self._precomputed_query_index is None:
            raise ValueError(
                'Query augmentation requires a compact .npz query archive.')
        if not pool_path:
            raise ValueError(
                'query_augmentation_pool_path is required when beta > 0.')
        path = Path(pool_path)
        if not path.is_file():
            raise FileNotFoundError(
                'Query augmentation pool not found: {}'.format(path))
        with np.load(path, allow_pickle=False) as pool:
            required = {'feature_ids', 'counts'}
            if not required.issubset(pool.files):
                raise ValueError(
                    'Query augmentation pool requires feature_ids and counts.')
            feature_ids = np.array(pool['feature_ids'], dtype=np.int64,
                                   copy=True)
            counts = np.array(pool['counts'], dtype=np.int64, copy=True)
        if feature_ids.ndim != 2:
            raise ValueError('Pool feature_ids must have shape [items, slots].')
        if counts.shape != (feature_ids.shape[0],):
            raise ValueError('Pool counts must have one value per item.')
        if feature_ids.shape[0] != const.item_id_num:
            raise ValueError(
                'Pool item rows {} do not match item vocabulary size {}.'.format(
                    feature_ids.shape[0], const.item_id_num))
        if (counts < 0).any() or (counts > feature_ids.shape[1]).any():
            raise ValueError('Pool counts contain an invalid slot count.')
        if feature_ids.size and (
                feature_ids.min() < 0
                or feature_ids.max() >= len(self._precomputed_query_features)):
            raise ValueError('Pool contains an invalid query feature ID.')
        slots = np.arange(feature_ids.shape[1])[None, :]
        if (feature_ids[slots >= counts[:, None]] != 0).any():
            raise ValueError('Unused query-pool slots must be zero padding.')
        if (feature_ids[slots < counts[:, None]] <= 0).any():
            raise ValueError('Active query-pool slots must be non-padding.')

        self.item_query_feature_ids = torch.from_numpy(feature_ids)
        self.item_query_feature_counts = torch.from_numpy(counts)

    def _lookup_precomputed_feature_ids(
            self, feature_ids: torch.Tensor,
            device: torch.device) -> torch.Tensor:
        cpu_ids = feature_ids.detach().to(
            device='cpu', dtype=torch.long).numpy()
        if cpu_ids.size and (
                cpu_ids.min() < 0
                or cpu_ids.max() >= self._precomputed_query_features.shape[0]):
            raise IndexError(
                'A feature ID is outside the precomputed query table.')
        values = np.array(
            self._precomputed_query_features[cpu_ids],
            dtype=np.float32,
            copy=True,
        )
        query = torch.from_numpy(values).to(device=device)
        if not torch.isfinite(query).all():
            raise ValueError(
                'Selected precomputed query features contain NaN/Inf.')
        if self.normalize_precomputed_queries:
            query = F.normalize(query, p=2, dim=-1)
        return query

    def _lookup_precomputed_queries(self, query_ids: torch.Tensor,
                                    device: torch.device) -> torch.Tensor:
        if query_ids is None:
            raise ValueError(
                'query_id is required for query_backend=precomputed_qwen.')
        cpu_ids = query_ids.detach().to(
            device='cpu', dtype=torch.long).numpy()
        if cpu_ids.size and (
                cpu_ids.min() < 0
                or cpu_ids.max() >= (
                    len(self._precomputed_query_index)
                    if self._precomputed_query_index is not None
                    else self._precomputed_query_features.shape[0])):
            raise IndexError(
                'A query_id is outside the precomputed feature table.')
        if self._precomputed_query_index is not None:
            feature_ids = self._precomputed_query_index[cpu_ids]
        else:
            feature_ids = cpu_ids
        return self._lookup_precomputed_feature_ids(
            torch.from_numpy(np.asarray(feature_ids)), device=device)

    def encode_query(self, token_ids: torch.Tensor | None,
                     query_ids: torch.Tensor = None) -> torch.Tensor:
        if self.query_backend == 'token':
            if token_ids is None:
                raise ValueError("token_ids is required for query_backend=token.")
            query = self.session_embedding.get_query_emb(token_ids)
        elif self.query_backend == 'query_id':
            if query_ids is None:
                raise ValueError(
                    'query_id is required for query_backend=query_id.')
            if query_ids.numel() and (
                    query_ids.min() < 0
                    or query_ids.max() >= self.session_query_ids.numel()):
                raise IndexError('A query_id is outside the session table.')
            table_ids = self.session_query_ids[query_ids.to(torch.long)]
            query = self.query_id_embedding(table_ids)
        else:
            if query_ids is None:
                raise ValueError(
                    'query_id is required for query_backend=precomputed_qwen.')
            query = self._lookup_precomputed_queries(
                query_ids, device=query_ids.device)
        return self.query_projection(query)

    def _search_history(self, session_ids: torch.Tensor):
        if session_ids.numel() and (
                session_ids.min() < 0
                or session_ids.max() >= self.session_map_vocab['pos_items'].size(0)):
            raise IndexError("A session_id is outside the session vocabulary.")
        positive_items = self.session_map_vocab['pos_items'][session_ids]
        if self.query_backend == 'precomputed_qwen':
            query_embedding = self.encode_query(None, session_ids)
        else:
            if 'keyword' not in self.session_map_vocab:
                raise RuntimeError(
                    "keyword table is required for query_backend={}. "
                    "The lightweight session vocabulary is valid only for "
                    "precomputed_qwen.".format(self.query_backend))
            keyword_ids = self.session_map_vocab['keyword'][session_ids]
            query_embedding = self.encode_query(keyword_ids, session_ids)
        positive_item_embedding = self.item_projection(
            self.session_embedding.get_item_emb(positive_items))
        item_mask = positive_items.ne(0)
        if item_mask.sum(dim=-1).gt(1).any():
            raise ValueError(
                'Multi-item search sessions must be expanded into '
                'query-item events before IntSR history encoding.')
        denominator = item_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        pooled_item_embedding = (
            positive_item_embedding
            * item_mask.unsqueeze(-1).to(positive_item_embedding.dtype)
        ).sum(dim=-2) / denominator

        first_index = item_mask.to(torch.int64).argmax(dim=-1, keepdim=True)
        representative_item = positive_items.gather(
            dim=-1, index=first_index).squeeze(-1)
        representative_item = representative_item.masked_fill(
            ~item_mask.any(dim=-1), 0)
        return query_embedding, pooled_item_embedding, representative_item

    def _sample_augmented_queries(self, item_ids: torch.Tensor,
                                  eligible: torch.Tensor):
        augmented = torch.zeros(
            (*item_ids.shape, self.hidden_dim),
            dtype=self.universal_query.dtype,
            device=item_ids.device,
        )
        selected = torch.zeros_like(eligible)
        if (not self.training
                or self.query_augmentation_beta <= 0
                or self.item_query_feature_ids is None):
            return augmented, selected

        counts = self.item_query_feature_counts[item_ids]
        selected = eligible & counts.gt(0)
        if self.query_augmentation_beta < 1.0:
            selected = selected & (
                torch.rand(
                    item_ids.shape,
                    device=item_ids.device,
                    generator=self._query_augmentation_generator,
                ) < self.query_augmentation_beta)
        if not selected.any():
            return augmented, selected

        selected_counts = counts[selected]
        slots = torch.floor(
            torch.rand(
                selected_counts.shape,
                device=item_ids.device,
                generator=self._query_augmentation_generator,
            ) * selected_counts.to(torch.float32)
        ).to(torch.long)
        feature_ids = self.item_query_feature_ids[
            item_ids[selected], slots]
        query = self._lookup_precomputed_feature_ids(
            feature_ids, device=item_ids.device)
        augmented[selected] = self.query_projection(query)
        return augmented, selected

    def _build_event_inputs(self, all_his: torch.Tensor,
                            all_his_type: torch.Tensor,
                            all_his_times: torch.Tensor):
        event_valid = all_his.ne(0)
        event_is_search = all_his_type.eq(2) & event_valid
        event_is_recommendation = all_his_type.eq(1) & event_valid

        recommendation_ids = all_his.masked_fill(~event_is_recommendation, 0)
        recommendation_item = self.item_projection(
            self.session_embedding.get_item_emb(recommendation_ids))

        session_ids = all_his.masked_fill(~event_is_search, 0)
        search_query, search_item, search_item_ids = self._search_history(
            session_ids)
        event_item = torch.where(
            event_is_search.unsqueeze(-1), search_item, recommendation_item)
        event_item_ids = torch.where(
            event_is_search, search_item_ids, recommendation_ids)

        universal_query = self.universal_query.reshape(1, 1, -1).expand(
            all_his.size(0), all_his.size(1), -1)
        augmented_query, query_augmented = self._sample_augmented_queries(
            recommendation_ids, event_is_recommendation)
        recommendation_query = torch.where(
            query_augmented.unsqueeze(-1), augmented_query, universal_query)
        event_query = torch.where(
            event_is_search.unsqueeze(-1), search_query, recommendation_query)
        if self.historical_query_include_item:
            event_query = event_query + event_item
        query_valid = event_is_search | query_augmented

        finite_times = torch.isfinite(all_his_times)
        safe_times = torch.where(
            finite_times,
            all_his_times,
            torch.zeros_like(all_his_times),
        ).to(torch.long)
        day_bucket = torch.div(safe_times, 86400,
                               rounding_mode='floor').remainder(
                                   self.absolute_time_buckets)
        scenario_token = self.absolute_time_embedding(day_bucket)

        # Public artifacts retain positive actions but not feedback subtypes.
        # Use one positive F token; S/R is represented by the DSF task tag.
        feedback_type = event_valid.to(all_his_type.dtype)
        feedback_token = self.feedback_embedding(feedback_type)

        # X1 contains S/I/F in chronological action order. X2 contains one Q
        # per historical action; the current request Q is appended later.
        ordinary = torch.stack(
            [scenario_token, event_item, feedback_token], dim=2)
        token_types = torch.tensor([0, 2, 3], device=ordinary.device)
        ordinary = ordinary + self.token_type_embedding(token_types).reshape(
            1, 1, 3, -1)
        ordinary = ordinary.reshape(all_his.size(0), -1, self.hidden_dim)
        event_query = event_query + self.token_type_embedding.weight[1]

        return {
            'ordinary': ordinary,
            'query': event_query,
            'valid': event_valid,
            'is_search': event_is_search,
            'query_valid': query_valid,
            'query_augmented': query_augmented,
            'times': safe_times,
            'item_ids': event_item_ids,
        }

    def _current_query_base(self, inputs):
        batch_size = inputs['user'].size(0)
        if inputs['search']:
            return self.encode_query(
                inputs['query'], inputs.get('query_id'))
        if self.prediction_mode == 'retrieval_dot':
            return self.universal_query.reshape(1, -1).expand(
                batch_size, -1)
        # Recommendation ranking replaces the Q placeholder with the
        # candidate item. The universal token is reserved for recommendation
        # retrieval, where no candidate item is supplied.
        return torch.zeros(
            batch_size,
            self.hidden_dim,
            dtype=self.universal_query.dtype,
            device=inputs['user'].device,
        )

    def _candidate_ids(self, inputs):
        if 'candidate_items' in inputs:
            return inputs['candidate_items']
        return torch.cat([
            inputs['item'].unsqueeze(1),
            inputs['neg_items'],
        ], dim=1)

    def _candidate_item_embeddings(self, candidate_ids):
        return self.item_projection(
            self.session_embedding.get_item_emb(candidate_ids))


    def encode_candidates(self, inputs, candidate_ids):
        all_his = inputs['all_his']
        all_his_type = inputs['all_his_type']
        all_his_times = inputs['all_his_ts']
        current_times = inputs['timestamp'].reshape(-1).to(torch.long)

        events = self._build_event_inputs(
            all_his, all_his_type, all_his_times)
        candidate_items = self._candidate_item_embeddings(candidate_ids)
        current_query_base = self._current_query_base(inputs).unsqueeze(1)
        if self.prediction_mode == 'retrieval_dot':
            candidate_queries = (
                current_query_base
                + self.token_type_embedding.weight[1].reshape(1, 1, -1)
            )
        else:
            candidate_queries = (
                current_query_base
                + candidate_items
                + self.token_type_embedding.weight[1].reshape(1, 1, -1)
            )

        event_group_ids = inputs.get('all_his_group_id')
        current_group_ids = inputs.get('current_group_id')
        if (event_group_ids is None) != (current_group_ids is None):
            raise ValueError('IntSR group tensors must be provided together')
        if event_group_ids is not None:
            event_group_ids = torch.as_tensor(event_group_ids, dtype=torch.long, device=all_his.device)
            current_group_ids = torch.as_tensor(current_group_ids, dtype=torch.long, device=all_his.device).reshape(-1)
            if event_group_ids.shape != all_his.shape:
                raise ValueError('all_his_group_id must have shape [batch, history_len]')
            if current_group_ids.shape != (all_his.size(0),):
                raise ValueError('current_group_id must have shape [batch]')
        masks = build_intsr_masks(
            event_valid=events['valid'],
            event_times=events['times'],
            event_item_ids=events['item_ids'],
            event_is_search=events['query_valid'],
            current_times=current_times,
            current_item_ids=(
                candidate_ids
                if self.prediction_mode == 'ranking_mlp'
                else None
            ),
            event_group_ids=event_group_ids,
            current_group_ids=current_group_ids,
        )
        current_day = torch.div(current_times, 86400,
                                rounding_mode='floor').remainder(
                                    self.absolute_time_buckets)
        current_scenario = self.absolute_time_embedding(current_day)
        current_s_token = (
            current_scenario + self.token_type_embedding.weight[0]
        ).unsqueeze(1)
        ordinary = torch.cat([events['ordinary'], current_s_token], dim=1)
        ordinary_times = torch.cat([
            events['times'].repeat_interleave(3, dim=1),
            current_times.unsqueeze(1),
        ], dim=1)
        _, _, candidate_interests = self.qdb.forward_candidates(
            ordinary,
            events['query'],
            candidate_queries,
            masks,
            ordinary_times,
            events['times'],
            current_times.unsqueeze(1),
        )

        user = self.user_projection(
            self.session_embedding.get_user_emb(inputs['user']))
        task_ids = torch.full_like(
            current_times, 2 if inputs['search'] else 1)
        task = (
            self.task_embedding(task_ids)
            if self.separate_task_embedding
            else self.feedback_embedding(task_ids))
        scenario_features = torch.cat([user, current_scenario, task], dim=-1)
        num_candidates = candidate_queries.size(1)
        expanded_scenarios = scenario_features.unsqueeze(1).expand(
            -1, num_candidates, -1)
        candidate_interests = self.dsf(
            candidate_interests.reshape(-1, self.hidden_dim),
            expanded_scenarios.reshape(-1, expanded_scenarios.size(-1)),
        ).reshape(candidate_ids.size(0), num_candidates, self.hidden_dim)
        return self.output_norm(candidate_interests), candidate_items

    def encode(self, inputs):
        positive_ids = inputs['item'].reshape(-1, 1)
        interests, _ = self.encode_candidates(inputs, positive_ids)
        return interests[:, 0]

    def score_candidates(self, interests, candidate_items=None):
        if self.prediction_mode == 'retrieval_dot':
            if candidate_items is None:
                raise ValueError(
                    'candidate_items are required for retrieval_dot.')
            return torch.matmul(
                interests, candidate_items.transpose(-2, -1)
            ).squeeze(1)
        return self.candidate_scorer(interests).squeeze(-1)

    def _loss(self, inputs):
        candidate_ids = self._candidate_ids(inputs)
        interests, candidate_items = self.encode_candidates(
            inputs, candidate_ids)
        logits = self.score_candidates(interests, candidate_items)
        labels = torch.zeros(logits.size(0), dtype=torch.long,
                             device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return {
            'click_loss': loss.detach().clone(),
            'total_loss': loss,
        }

    def _predict(self, inputs):
        candidate_ids = self._candidate_ids(inputs)
        interests, candidate_items = self.encode_candidates(
            inputs, candidate_ids)
        return self.score_candidates(interests, candidate_items)

    def rec_loss(self, inputs):
        return self._loss(inputs)

    def src_loss(self, inputs):
        return self._loss(inputs)

    def rec_predict(self, inputs):
        return self._predict(inputs)

    def src_predict(self, inputs):
        return self._predict(inputs)

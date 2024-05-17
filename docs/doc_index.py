

repos/datasets/benchmarks/benchmark_array_xd.py
-------------------------functions----------------------
benchmark_array_xd()
read_batch_formatted_as_numpy(feats, tmp_dir)
read_batch_unformated(feats, tmp_dir)
read_col_formatted_as_numpy(feats, tmp_dir)
read_col_unformated(feats, tmp_dir)
read_formatted_as_numpy(feats, tmp_dir)
read_unformated(feats, tmp_dir)
write(my_features, dummy_data, tmp_dir)



repos/datasets/benchmarks/benchmark_getitem_100B.py
-------------------------functions----------------------
benchmark_table_100B()
generate_100B_dataset(num_examples: int, chunk_size: int)
get_batch_of_1024_random_rows(dataset: datasets.Dataset)
get_batch_of_1024_rows(dataset: datasets.Dataset)
get_first_row(dataset: datasets.Dataset)
get_last_row(dataset: datasets.Dataset)

-------------------------methods----------------------
RandIter.__iter__(self)
RandIter.__len__(self)
RandIter.__post_init__(self)


repos/datasets/benchmarks/benchmark_indices_mapping.py
-------------------------functions----------------------
benchmark_indices_mapping()
select(dataset: datasets.Dataset)
shard(dataset: datasets.Dataset, num_shards = 10)
shuffle(dataset: datasets.Dataset)
sort(dataset: datasets.Dataset)
train_test_split(dataset: datasets.Dataset)



repos/datasets/benchmarks/benchmark_iterating.py
-------------------------functions----------------------
benchmark_iterating()
read(dataset: datasets.Dataset, length)
read_batch(dataset: datasets.Dataset, length, batch_size)
read_formatted(dataset: datasets.Dataset, length, type)
read_formatted_batch(dataset: datasets.Dataset, length, batch_size, type)



repos/datasets/benchmarks/benchmark_map_filter.py
-------------------------functions----------------------
benchmark_map_filter()
filter(dataset: datasets.Dataset, **kwargs)
map(dataset: datasets.Dataset, **kwargs)



repos/datasets/benchmarks/format.py
-------------------------functions----------------------
format_json_to_md(input_json_file, output_md_file)



repos/datasets/benchmarks/utils.py
-------------------------functions----------------------
generate_example_dataset(dataset_path, features, num_examples = 100, seq_shapes = None)
generate_examples(features: dict, num_examples = 100, seq_shapes = None)
get_duration(func)



repos/datasets/docs/source/_config.py


repos/datasets/metrics/accuracy/accuracy.py
-------------------------methods----------------------
Accuracy._compute(self, predictions, references, normalize = True, sample_weight = None)
Accuracy._info(self)


repos/datasets/metrics/bertscore/bertscore.py
-------------------------functions----------------------
filter_logging_context()

-------------------------methods----------------------
BERTScore._compute(self, predictions, references, lang = None, model_type = None, num_layers = None, verbose = False, idf = False, device = None, batch_size = 64, nthreads = 4, all_layers = False, rescale_with_baseline = False, baseline_path = None, use_fast_tokenizer = False, )
BERTScore._info(self)
BERTScore.add(self, prediction = None, reference = None, **kwargs)
BERTScore.add_batch(self, predictions = None, references = None, **kwargs)


repos/datasets/metrics/bleu/bleu.py
-------------------------methods----------------------
Bleu._compute(self, predictions, references, max_order = 4, smooth = False)
Bleu._info(self)


repos/datasets/metrics/bleurt/bleurt.py
-------------------------methods----------------------
BLEURT._compute(self, predictions, references)
BLEURT._download_and_prepare(self, dl_manager)
BLEURT._info(self)


repos/datasets/metrics/cer/cer.py
-------------------------methods----------------------
CER._compute(self, predictions, references, concatenate_texts = False)
CER._info(self)


repos/datasets/metrics/cer/test_cer.py
-------------------------methods----------------------
TestCER.test_cer_case_senstive(self)
TestCER.test_cer_del(self)
TestCER.test_cer_empty(self)
TestCER.test_cer_equal(self)
TestCER.test_cer_insert(self)
TestCER.test_cer_list_of_seqs(self)
TestCER.test_cer_sub(self)
TestCER.test_cer_unicode(self)
TestCER.test_cer_whitespace(self)
TestCER.test_correlated_sentences(self)


repos/datasets/metrics/chrf/chrf.py
-------------------------methods----------------------
ChrF._compute(self, predictions, references, char_order: int  =  CHRF.CHAR_ORDER, word_order: int  =  CHRF.WORD_ORDER, beta: int  =  CHRF.BETA, lowercase: bool  =  False, whitespace: bool  =  False, eps_smoothing: bool  =  False, )
ChrF._info(self)


repos/datasets/metrics/code_eval/code_eval.py
-------------------------functions----------------------
estimate_pass_at_k(num_samples, num_correct, k)

-------------------------methods----------------------
CodeEval._compute(self, predictions, references, k = [1, 10, 100], num_workers = 4, timeout = 3.0)
CodeEval._info(self)


repos/datasets/metrics/code_eval/execute.py
-------------------------functions----------------------
chdir(root)
check_correctness(check_program, timeout, task_id, completion_id)
create_tempdir()
reliability_guard(maximum_memory_bytes = None)
swallow_io()
time_limit(seconds)
unsafe_execute(check_program, result, timeout)

-------------------------methods----------------------
WriteOnlyStringIO.read(self, *args, **kwargs)
WriteOnlyStringIO.readable(self, *args, **kwargs)
WriteOnlyStringIO.readline(self, *args, **kwargs)
WriteOnlyStringIO.readlines(self, *args, **kwargs)


repos/datasets/metrics/comet/comet.py
-------------------------methods----------------------
COMET._compute(self, sources, predictions, references, gpus = None, progress_bar = False)
COMET._download_and_prepare(self, dl_manager)
COMET._info(self)


repos/datasets/metrics/competition_math/competition_math.py


repos/datasets/metrics/coval/coval.py
-------------------------functions----------------------
check_gold_parse_annotation(key_lines)
evaluate(key_lines, sys_lines, metrics, NP_only, remove_nested, keep_singletons, min_span)
get_coref_infos(key_lines, sys_lines, NP_only = False, remove_nested = False, keep_singletons = True, min_span = False, doc = "dummy_doc")

-------------------------methods----------------------
Coval._compute(self, predictions, references, keep_singletons = True, NP_only = False, min_span = False, remove_nested = False)
Coval._info(self)


repos/datasets/metrics/cuad/cuad.py
-------------------------methods----------------------
CUAD._compute(self, predictions, references)
CUAD._info(self)


repos/datasets/metrics/cuad/evaluate.py
-------------------------functions----------------------
compute_precision_recall(predictions, ground_truths, qa_id)
evaluate(dataset, predictions)
exact_match_score(prediction, ground_truth)
get_aupr(precisions, recalls)
get_jaccard(prediction, ground_truth)
get_prec_at_recall(precisions, recalls, recall_thresh)
metric_max_over_ground_truths(metric_fn, predictions, ground_truths)
normalize_answer(s)
process_precisions(precisions)



repos/datasets/metrics/exact_match/exact_match.py
-------------------------methods----------------------
ExactMatch._compute(self, predictions, references, regexes_to_ignore = None, ignore_case = False, ignore_punctuation = False, ignore_numbers = False, )
ExactMatch._info(self)


repos/datasets/metrics/f1/f1.py
-------------------------methods----------------------
F1._compute(self, predictions, references, labels = None, pos_label = 1, average = "binary", sample_weight = None)
F1._info(self)


repos/datasets/metrics/frugalscore/frugalscore.py
-------------------------methods----------------------
FRUGALSCORE._compute(self, predictions, references, batch_size = 32, max_length = 128, device = None, )
FRUGALSCORE._download_and_prepare(self, dl_manager)
FRUGALSCORE._info(self)


repos/datasets/metrics/glue/glue.py
-------------------------functions----------------------
acc_and_f1(preds, labels)
pearson_and_spearman(preds, labels)
simple_accuracy(preds, labels)

-------------------------methods----------------------
Glue._compute(self, predictions, references)
Glue._info(self)


repos/datasets/metrics/google_bleu/google_bleu.py
-------------------------methods----------------------
GoogleBleu._compute(self, predictions: List[List[List[str]]], references: List[List[str]], min_len: int  =  1, max_len: int  =  4, )
GoogleBleu._info(self)


repos/datasets/metrics/indic_glue/indic_glue.py
-------------------------functions----------------------
acc_and_f1(preds, labels)
precision_at_10(en_sentvecs, in_sentvecs)
simple_accuracy(preds, labels)

-------------------------methods----------------------
IndicGlue._compute(self, predictions, references)
IndicGlue._info(self)


repos/datasets/metrics/mae/mae.py
-------------------------methods----------------------
Mae._compute(self, predictions, references, sample_weight = None, multioutput = "uniform_average")
Mae._get_feature_types(self)
Mae._info(self)


repos/datasets/metrics/mahalanobis/mahalanobis.py
-------------------------methods----------------------
Mahalanobis._compute(self, X, reference_distribution)
Mahalanobis._info(self)


repos/datasets/metrics/matthews_correlation/matthews_correlation.py
-------------------------methods----------------------
MatthewsCorrelation._compute(self, predictions, references, sample_weight = None)
MatthewsCorrelation._info(self)


repos/datasets/metrics/mauve/mauve.py
-------------------------methods----------------------
Mauve._compute(self, predictions, references, p_features = None, q_features = None, p_tokens = None, q_tokens = None, num_buckets = "auto", pca_max_data = -1, kmeans_explained_var = 0.9, kmeans_num_redo = 5, kmeans_max_iter = 500, featurize_model_name = "gpt2-large", device_id = -1, max_text_length = 1024, divergence_curve_discretization_size = 25, mauve_scaling_factor = 5, verbose = True, seed = 25, )
Mauve._info(self)


repos/datasets/metrics/mean_iou/mean_iou.py
-------------------------functions----------------------
intersect_and_union(pred_label, label, num_labels, ignore_index: bool, label_map: Optional[Dict[int, int]]  =  None, reduce_labels: bool  =  False, )
mean_iou(results, gt_seg_maps, num_labels, ignore_index: bool, nan_to_num: Optional[int]  =  None, label_map: Optional[Dict[int, int]]  =  None, reduce_labels: bool  =  False, )
total_intersect_and_union(results, gt_seg_maps, num_labels, ignore_index: bool, label_map: Optional[Dict[int, int]]  =  None, reduce_labels: bool  =  False, )

-------------------------methods----------------------
MeanIoU._compute(self, predictions, references, num_labels: int, ignore_index: bool, nan_to_num: Optional[int]  =  None, label_map: Optional[Dict[int, int]]  =  None, reduce_labels: bool  =  False, )
MeanIoU._info(self)


repos/datasets/metrics/meteor/meteor.py
-------------------------methods----------------------
Meteor._compute(self, predictions, references, alpha = 0.9, beta = 3, gamma = 0.5)
Meteor._download_and_prepare(self, dl_manager)
Meteor._info(self)


repos/datasets/metrics/mse/mse.py
-------------------------methods----------------------
Mse._compute(self, predictions, references, sample_weight = None, multioutput = "uniform_average", squared = True)
Mse._get_feature_types(self)
Mse._info(self)


repos/datasets/metrics/pearsonr/pearsonr.py
-------------------------methods----------------------
Pearsonr._compute(self, predictions, references, return_pvalue = False)
Pearsonr._info(self)


repos/datasets/metrics/perplexity/perplexity.py
-------------------------methods----------------------
Perplexity._compute(self, input_texts, model_id, batch_size: int  =  16, add_start_token: bool  =  True, device = None)
Perplexity._info(self)


repos/datasets/metrics/precision/precision.py
-------------------------methods----------------------
Precision._compute(self, predictions, references, labels = None, pos_label = 1, average = "binary", sample_weight = None, zero_division = "warn", )
Precision._info(self)


repos/datasets/metrics/recall/recall.py
-------------------------methods----------------------
Recall._compute(self, predictions, references, labels = None, pos_label = 1, average = "binary", sample_weight = None, zero_division = "warn", )
Recall._info(self)


repos/datasets/metrics/roc_auc/roc_auc.py
-------------------------methods----------------------
ROCAUC._compute(self, references, prediction_scores, average = "macro", sample_weight = None, max_fpr = None, multi_class = "raise", labels = None, )
ROCAUC._info(self)


repos/datasets/metrics/rouge/rouge.py
-------------------------methods----------------------
Rouge._compute(self, predictions, references, rouge_types = None, use_aggregator = True, use_stemmer = False)
Rouge._info(self)


repos/datasets/metrics/sacrebleu/sacrebleu.py
-------------------------methods----------------------
Sacrebleu._compute(self, predictions, references, smooth_method = "exp", smooth_value = None, force = False, lowercase = False, tokenize = None, use_effective_order = False, )
Sacrebleu._info(self)


repos/datasets/metrics/sari/sari.py
-------------------------functions----------------------
SARIngram(sgrams, cgrams, rgramslist, numref)
SARIsent(ssent, csent, rsents)
normalize(sentence, lowercase: bool  =  True, tokenizer: str  =  "13a", return_str: bool  =  True)

-------------------------methods----------------------
Sari._compute(self, sources, predictions, references)
Sari._info(self)


repos/datasets/metrics/seqeval/seqeval.py
-------------------------methods----------------------
Seqeval._compute(self, predictions, references, suffix: bool  =  False, scheme: Optional[str]  =  None, mode: Optional[str]  =  None, sample_weight: Optional[List[int]]  =  None, zero_division: Union[str, int]  =  "warn", )
Seqeval._info(self)


repos/datasets/metrics/spearmanr/spearmanr.py


repos/datasets/metrics/squad/evaluate.py
-------------------------functions----------------------
evaluate(dataset, predictions)
exact_match_score(prediction, ground_truth)
f1_score(prediction, ground_truth)
metric_max_over_ground_truths(metric_fn, prediction, ground_truths)
normalize_answer(s)



repos/datasets/metrics/squad/squad.py
-------------------------methods----------------------
Squad._compute(self, predictions, references)
Squad._info(self)


repos/datasets/metrics/squad_v2/evaluate.py
-------------------------functions----------------------
apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh)
compute_exact(a_gold, a_pred)
compute_f1(a_gold, a_pred)
find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
find_best_thresh(preds, scores, na_probs, qid_to_has_ans)
get_raw_scores(dataset, preds)
get_tokens(s)
histogram_na_prob(na_probs, qid_list, image_dir, name)
main()
make_eval_dict(exact_scores, f1_scores, qid_list = None)
make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans, out_image = None, title = None)
make_qid_to_has_ans(dataset)
merge_eval(main_eval, new_eval, prefix)
normalize_answer(s)
parse_args()
plot_pr_curve(precisions, recalls, out_image, title)
run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs, qid_to_has_ans, out_image_dir)



repos/datasets/metrics/squad_v2/squad_v2.py
-------------------------methods----------------------
SquadV2._compute(self, predictions, references, no_answer_threshold = 1.0)
SquadV2._info(self)


repos/datasets/metrics/super_glue/record_evaluation.py
-------------------------functions----------------------
evaluate(dataset, predictions)
exact_match_score(prediction, ground_truth)
f1_score(prediction, ground_truth)
metric_max_over_ground_truths(metric_fn, prediction, ground_truths)
normalize_answer(s)



repos/datasets/metrics/super_glue/super_glue.py
-------------------------functions----------------------
acc_and_f1(preds, labels, f1_avg = "binary")
evaluate_multirc(ids_preds, labels)
simple_accuracy(preds, labels)

-------------------------methods----------------------
SuperGlue._compute(self, predictions, references)
SuperGlue._get_feature_types(self)
SuperGlue._info(self)


repos/datasets/metrics/ter/ter.py
-------------------------methods----------------------
Ter._compute(self, predictions, references, normalized: bool  =  False, ignore_punct: bool  =  False, support_zh_ja_chars: bool  =  False, case_sensitive: bool  =  False, )
Ter._info(self)


repos/datasets/metrics/wer/wer.py
-------------------------methods----------------------
WER._compute(self, predictions = None, references = None, concatenate_texts = False)
WER._info(self)


repos/datasets/metrics/wiki_split/wiki_split.py
-------------------------functions----------------------
SARIngram(sgrams, cgrams, rgramslist, numref)
SARIsent(ssent, csent, rsents)
compute_em(predictions, references)
compute_exact(a_gold, a_pred)
compute_sacrebleu(predictions, references, smooth_method = "exp", smooth_value = None, force = False, lowercase = False, use_effective_order = False, )
compute_sari(sources, predictions, references)
normalize(sentence, lowercase: bool  =  True, tokenizer: str  =  "13a", return_str: bool  =  True)
normalize_answer(s)

-------------------------methods----------------------
WikiSplit._compute(self, sources, predictions, references)
WikiSplit._info(self)


repos/datasets/metrics/xnli/xnli.py
-------------------------functions----------------------
simple_accuracy(preds, labels)

-------------------------methods----------------------
Xnli._compute(self, predictions, references)
Xnli._info(self)


repos/datasets/metrics/xtreme_s/xtreme_s.py
-------------------------functions----------------------
bleu(preds, labels, smooth_method = "exp", smooth_value = None, force = False, lowercase = False, tokenize = None, use_effective_order = False, )
f1_and_simple_accuracy(preds, labels)
simple_accuracy(preds, labels)
wer_and_cer(preds, labels, concatenate_texts, config_name)

-------------------------methods----------------------
XtremeS._compute(self, predictions, references, bleu_kwargs = None, wer_kwargs = None)
XtremeS._info(self)


repos/datasets/setup.py


repos/datasets/src/datasets/__init__.py


repos/datasets/src/datasets/arrow_dataset.py
-------------------------functions----------------------
_check_column_names(column_names: List[str])
_check_table(table)
_check_valid_indices_value(index, size)
_concatenate_map_style_datasets(dsets: List[Dataset], info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, axis: int  =  0, )
_interleave_map_style_datasets(datasets: List["Dataset"], probabilities: Optional[List[float]]  =  None, seed: Optional[int]  =  None, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, stopping_strategy: Literal["first_exhausted", "all_exhausted"]  =  "first_exhausted", **kwargs, )
_split_by_node_map_style_dataset(dataset: Dataset, rank: int, world_size: int)
get_indices_from_mask_function(function: Callable, batched: bool, with_indices: bool, with_rank: bool, input_columns: Optional[Union[str, List[str]]], indices_mapping: Optional[Table]  =  None, *args, **fn_kwargs, )
transmit_format(func)
transmit_tasks(func)
update_metadata_with_features(table: Table, features: Features)

-------------------------methods----------------------
Dataset.__del__(self)
Dataset.__enter__(self)
Dataset.__exit__(self, exc_type, exc_val, exc_tb)
Dataset.__getitem__(self, key: Union[int, slice, Iterable[int]]) -> Dict:  # noqa: F811...@overloadself, key: str) -> List:  # noqa: F811...self, key):  # noqa: F811key)self, keys: List) -> List:)
Dataset.__getitem__(self, key: Union[int, slice, Iterable[int]]) -> Dict:  # noqa: F811...@overloadself, key: str) -> List:  # noqa: F811...self, key):  # noqa: F811key)self, keys: List) -> List:)
Dataset.__getitem__(self, key: Union[int, slice, Iterable[int]]) -> Dict:  # noqa: F811...@overloadself, key: str) -> List:  # noqa: F811...self, key):  # noqa: F811key)self, keys: List) -> List:)
Dataset.__getitems__(self, keys: List)
Dataset.__init__(self, arrow_table: Table, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, indices_table: Optional[Table]  =  None, fingerprint: Optional[str]  =  None, )
Dataset.__iter__(self)
Dataset.__len__(self)
Dataset.__repr__(self)
Dataset.__setstate__(self, state)
Dataset._build_local_temp_path(uri_or_path: str)
Dataset._estimate_nbytes(self)
Dataset._generate_tables_from_cache_file(filename: str)
Dataset._generate_tables_from_shards(shards: List["Dataset"], batch_size: int)
Dataset._get_cache_file_path(self, fingerprint)
Dataset._getitem(self, key: Union[int, slice, str, ListLike[int]], **kwargs)
Dataset._map_single(shard: "Dataset", function: Optional[Callable]  =  None, with_indices: bool  =  False, with_rank: bool  =  False, input_columns: Optional[List[str]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, drop_last_batch: bool  =  False, remove_columns: Optional[List[str]]  =  None, keep_in_memory: bool  =  False, cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, features: Optional[Features]  =  None, disable_nullable: bool  =  False, fn_kwargs: Optional[dict]  =  None, new_fingerprint: Optional[str]  =  None, rank: Optional[int]  =  None, offset: int  =  0, )
Dataset._new_dataset_with_indices(self, indices_cache_file_name: Optional[str]  =  None, indices_buffer: Optional[pa.Buffer]  =  None, fingerprint: Optional[str]  =  None, )
Dataset._push_parquet_shards_to_hub(self, repo_id: str, data_dir: str  =  "data", split: Optional[str]  =  None, token: Optional[str]  =  None, revision: Optional[str]  =  None, create_pr: Optional[bool]  =  False, max_shard_size: Optional[Union[int, str]]  =  None, num_shards: Optional[int]  =  None, embed_external_files: bool  =  True, )
Dataset._save_to_disk_single(job_id: int, shard: "Dataset", fpath: str, storage_options: Optional[dict])
Dataset._select_contiguous(self, start: int, length: int, new_fingerprint: Optional[str]  =  None, )
Dataset._select_with_indices_mapping(self, indices: Iterable, keep_in_memory: bool  =  False, indices_cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, new_fingerprint: Optional[str]  =  None, )
Dataset.add_column(self, name: str, column: Union[list, np.array], new_fingerprint: str)
Dataset.add_elasticsearch_index(self, column: str, index_name: Optional[str]  =  None, host: Optional[str]  =  None, port: Optional[int]  =  None, es_client: Optional["elasticsearch.Elasticsearch"]  =  None, # noqa: F821es_index_name: Optional[str]  =  None, es_index_config: Optional[dict]  =  None, )
Dataset.add_faiss_index(self, column: str, index_name: Optional[str]  =  None, device: Optional[int]  =  None, string_factory: Optional[str]  =  None, metric_type: Optional[int]  =  None, custom_index: Optional["faiss.Index"]  =  None, # noqa: F821batch_size: int  =  1000, train_size: Optional[int]  =  None, faiss_verbose: bool  =  False, dtype = np.float32, )
Dataset.add_faiss_index_from_external_arrays(self, external_arrays: np.array, index_name: str, device: Optional[int]  =  None, string_factory: Optional[str]  =  None, metric_type: Optional[int]  =  None, custom_index: Optional["faiss.Index"]  =  None, # noqa: F821batch_size: int  =  1000, train_size: Optional[int]  =  None, faiss_verbose: bool  =  False, dtype = np.float32, )
Dataset.add_item(self, item: dict, new_fingerprint: str)
Dataset.align_labels_with_mapping(self, label2id: Dict, label_column: str)
Dataset.cache_files(self)
Dataset.cast(self, features: Features, batch_size: Optional[int]  =  1000, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, num_proc: Optional[int]  =  None, )
Dataset.cast_column(self, column: str, feature: FeatureType, new_fingerprint: Optional[str]  =  None)
Dataset.class_encode_column(self, column: str, include_nulls: bool  =  False)
Dataset.cleanup_cache_files(self)
Dataset.column_names(self)
Dataset.data(self)
Dataset.export(self, filename: str, format: str  =  "tfrecord", )
Dataset.features(self)
Dataset.filter(self, function: Optional[Callable]  =  None, with_indices: bool  =  False, with_rank: bool  =  False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, fn_kwargs: Optional[dict]  =  None, num_proc: Optional[int]  =  None, suffix_template: str  =  "_{rank:05d}_of_{num_proc:05d}", new_fingerprint: Optional[str]  =  None, desc: Optional[str]  =  None, )
Dataset.flatten(self, new_fingerprint: Optional[str]  =  None, max_depth = 16)
Dataset.flatten_indices(self, keep_in_memory: bool  =  False, cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, features: Optional[Features]  =  None, disable_nullable: bool  =  False, num_proc: Optional[int]  =  None, new_fingerprint: Optional[str]  =  None, )
Dataset.format(self)
Dataset.formatted_as(self, type: Optional[str]  =  None, columns: Optional[List]  =  None, output_all_columns: bool  =  False, **format_kwargs, )
Dataset.from_buffer(cls, buffer: pa.Buffer, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, indices_buffer: Optional[pa.Buffer]  =  None, )
Dataset.from_csv(path_or_paths: Union[PathLike, List[PathLike]], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, num_proc: Optional[int]  =  None, **kwargs, )
Dataset.from_dict(cls, mapping: dict, features: Optional[Features]  =  None, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, )
Dataset.from_file(cls, filename: str, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, indices_filename: Optional[str]  =  None, in_memory: bool  =  False, )
Dataset.from_generator(generator: Callable, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, gen_kwargs: Optional[dict]  =  None, num_proc: Optional[int]  =  None, **kwargs, )
Dataset.from_json(path_or_paths: Union[PathLike, List[PathLike]], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, field: Optional[str]  =  None, num_proc: Optional[int]  =  None, **kwargs, )
Dataset.from_list(cls, mapping: List[dict], features: Optional[Features]  =  None, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, )
Dataset.from_pandas(cls, df: pd.DataFrame, features: Optional[Features]  =  None, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, preserve_index: Optional[bool]  =  None, )
Dataset.from_parquet(path_or_paths: Union[PathLike, List[PathLike]], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, columns: Optional[List[str]]  =  None, num_proc: Optional[int]  =  None, **kwargs, )
Dataset.from_polars(cls, df: "pl.DataFrame", features: Optional[Features]  =  None, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, )
Dataset.from_spark(df: "pyspark.sql.DataFrame", split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, keep_in_memory: bool  =  False, cache_dir: str  =  None, working_dir: str  =  None, load_from_cache_file: bool  =  True, **kwargs, )
Dataset.from_sql(sql: Union[str, "sqlalchemy.sql.Selectable"], con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"], features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, **kwargs, )
Dataset.from_text(path_or_paths: Union[PathLike, List[PathLike]], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, num_proc: Optional[int]  =  None, **kwargs, )
Dataset.iter(self, batch_size: int, drop_last_batch: bool  =  False)
Dataset.load_from_disk(dataset_path: str, fs = "deprecated", keep_in_memory: Optional[bool]  =  None, storage_options: Optional[dict]  =  None, )
Dataset.map(self, function: Optional[Callable]  =  None, with_indices: bool  =  False, with_rank: bool  =  False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, drop_last_batch: bool  =  False, remove_columns: Optional[Union[str, List[str]]]  =  None, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, features: Optional[Features]  =  None, disable_nullable: bool  =  False, fn_kwargs: Optional[dict]  =  None, num_proc: Optional[int]  =  None, suffix_template: str  =  "_{rank:05d}_of_{num_proc:05d}", new_fingerprint: Optional[str]  =  None, desc: Optional[str]  =  None, )
Dataset.num_columns(self)
Dataset.num_rows(self)
Dataset.prepare_for_task(self, task: Union[str, TaskTemplate], id: int  =  0)
Dataset.push_to_hub(self, repo_id: str, config_name: str  =  "default", set_default: Optional[bool]  =  None, split: Optional[str]  =  None, data_dir: Optional[str]  =  None, commit_message: Optional[str]  =  None, commit_description: Optional[str]  =  None, private: Optional[bool]  =  False, token: Optional[str]  =  None, revision: Optional[str]  =  None, branch = "deprecated", create_pr: Optional[bool]  =  False, max_shard_size: Optional[Union[int, str]]  =  None, num_shards: Optional[int]  =  None, embed_external_files: bool  =  True, )
Dataset.remove_columns(self, column_names: Union[str, List[str]], new_fingerprint: Optional[str]  =  None)
Dataset.rename_column(self, original_column_name: str, new_column_name: str, new_fingerprint: Optional[str]  =  None)
Dataset.rename_columns(self, column_mapping: Dict[str, str], new_fingerprint: Optional[str]  =  None)
Dataset.reset_format(self)
Dataset.save_to_disk(self, dataset_path: PathLike, fs = "deprecated", max_shard_size: Optional[Union[str, int]]  =  None, num_shards: Optional[int]  =  None, num_proc: Optional[int]  =  None, storage_options: Optional[dict]  =  None, )
Dataset.select(self, indices: Iterable, keep_in_memory: bool  =  False, indices_cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, new_fingerprint: Optional[str]  =  None, )
Dataset.select_columns(self, column_names: Union[str, List[str]], new_fingerprint: Optional[str]  =  None)
Dataset.set_format(self, type: Optional[str]  =  None, columns: Optional[List]  =  None, output_all_columns: bool  =  False, **format_kwargs, )
Dataset.set_transform(self, transform: Optional[Callable], columns: Optional[List]  =  None, output_all_columns: bool  =  False, )
Dataset.shape(self)
Dataset.shard(self, num_shards: int, index: int, contiguous: bool  =  False, keep_in_memory: bool  =  False, indices_cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, )
Dataset.shuffle(self, seed: Optional[int]  =  None, generator: Optional[np.random.Generator]  =  None, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, indices_cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, new_fingerprint: Optional[str]  =  None, )
Dataset.skip(self, n: int)
Dataset.sort(self, column_names: Union[str, Sequence_[str]], reverse: Union[bool, Sequence_[bool]]  =  False, kind = "deprecated", null_placement: str  =  "at_end", keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, indices_cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, new_fingerprint: Optional[str]  =  None, )
Dataset.take(self, n: int)
Dataset.to_csv(self, path_or_buf: Union[PathLike, BinaryIO], batch_size: Optional[int]  =  None, num_proc: Optional[int]  =  None, storage_options: Optional[dict]  =  None, **to_csv_kwargs, )
Dataset.to_dict(self, batch_size: Optional[int]  =  None, batched = "deprecated")
Dataset.to_iterable_dataset(self, num_shards: Optional[int]  =  1)
Dataset.to_json(self, path_or_buf: Union[PathLike, BinaryIO], batch_size: Optional[int]  =  None, num_proc: Optional[int]  =  None, storage_options: Optional[dict]  =  None, **to_json_kwargs, )
Dataset.to_list(self)
Dataset.to_pandas(self, batch_size: Optional[int]  =  None, batched: bool  =  False)
Dataset.to_parquet(self, path_or_buf: Union[PathLike, BinaryIO], batch_size: Optional[int]  =  None, storage_options: Optional[dict]  =  None, **parquet_writer_kwargs, )
Dataset.to_polars(self, batch_size: Optional[int]  =  None, batched: bool  =  False, schema_overrides: Optional[dict]  =  None, rechunk: bool  =  True, )
Dataset.to_sql(self, name: str, con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"], batch_size: Optional[int]  =  None, **sql_writer_kwargs, )
Dataset.train_test_split(self, test_size: Union[float, int, None]  =  None, train_size: Union[float, int, None]  =  None, shuffle: bool  =  True, stratify_by_column: Optional[str]  =  None, seed: Optional[int]  =  None, generator: Optional[np.random.Generator]  =  None, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, train_indices_cache_file_name: Optional[str]  =  None, test_indices_cache_file_name: Optional[str]  =  None, writer_batch_size: Optional[int]  =  1000, train_new_fingerprint: Optional[str]  =  None, test_new_fingerprint: Optional[str]  =  None, )
Dataset.unique(self, column: str)
Dataset.with_format(self, type: Optional[str]  =  None, columns: Optional[List]  =  None, output_all_columns: bool  =  False, **format_kwargs, )
Dataset.with_transform(self, transform: Optional[Callable], columns: Optional[List]  =  None, output_all_columns: bool  =  False, )
DatasetInfoMixin.__init__(self, info: DatasetInfo, split: Optional[NamedSplit])
DatasetInfoMixin.builder_name(self)
DatasetInfoMixin.citation(self)
DatasetInfoMixin.config_name(self)
DatasetInfoMixin.dataset_size(self)
DatasetInfoMixin.description(self)
DatasetInfoMixin.download_checksums(self)
DatasetInfoMixin.download_size(self)
DatasetInfoMixin.features(self)
DatasetInfoMixin.homepage(self)
DatasetInfoMixin.info(self)
DatasetInfoMixin.license(self)
DatasetInfoMixin.size_in_bytes(self)
DatasetInfoMixin.split(self)
DatasetInfoMixin.supervised_keys(self)
DatasetInfoMixin.task_templates(self)
DatasetInfoMixin.version(self)
TensorflowDatasetMixin._get_output_signature(dataset: "Dataset", collate_fn: Callable, collate_fn_args: dict, cols_to_retain: Optional[List[str]]  =  None, batch_size: Optional[int]  =  None, num_test_batches: int  =  20, )
TensorflowDatasetMixin.to_tf_dataset(self, batch_size: Optional[int]  =  None, columns: Optional[Union[str, List[str]]]  =  None, shuffle: bool  =  False, collate_fn: Optional[Callable]  =  None, drop_remainder: bool  =  False, collate_fn_args: Optional[Dict[str, Any]]  =  None, label_cols: Optional[Union[str, List[str]]]  =  None, prefetch: bool  =  True, num_workers: int  =  0, num_test_batches: int  =  20, )


repos/datasets/src/datasets/arrow_reader.py
-------------------------functions----------------------
_pct_to_abs_closest(boundary, num_examples)
_pct_to_abs_pct1(boundary, num_examples)
_rel_to_abs_instr(rel_instr, name2len)
_str_to_read_instruction(spec)
make_file_instructions(name: str, split_infos: List["SplitInfo"], instruction: Union[str, "ReadInstruction"], filetype_suffix: Optional[str]  =  None, prefix_path: Optional[str]  =  None, )

-------------------------methods----------------------
ArrowReader.__init__(self, path: str, info: Optional["DatasetInfo"])
ArrowReader._get_table_from_filename(self, filename_skip_take, in_memory = False)
ArrowReader.read_table(filename, in_memory = False)
BaseReader.__init__(self, path: str, info: Optional["DatasetInfo"])
BaseReader._get_table_from_filename(self, filename_skip_take, in_memory = False)
BaseReader._read_files(self, files, in_memory = False)
BaseReader.download_from_hf_gcs(self, download_config: DownloadConfig, relative_data_dir)
BaseReader.get_file_instructions(self, name, instruction, split_infos)
BaseReader.read(self, name, instructions, split_infos, in_memory = False, )
BaseReader.read_files(self, files: List[dict], original_instructions: Union[None, "ReadInstruction", "Split"]  =  None, in_memory = False, )
ParquetReader.__init__(self, path: str, info: Optional["DatasetInfo"])
ParquetReader._get_table_from_filename(self, filename_skip_take, **kwargs)
ReadInstruction.__add__(self, other)
ReadInstruction.__init__(self, split_name, rounding = None, from_ = None, to = None, unit = None)
ReadInstruction.__repr__(self)
ReadInstruction.__str__(self)
ReadInstruction._init(self, relative_instructions)
ReadInstruction._read_instruction_from_relative_instructions(cls, relative_instructions)
ReadInstruction.from_spec(cls, spec)
ReadInstruction.to_absolute(self, name2len)
ReadInstruction.to_spec(self)
_RelativeInstruction.__post_init__(self)


repos/datasets/src/datasets/arrow_writer.py
-------------------------functions----------------------
get_parquet_lengths(sources)
parquet_to_arrow(source, destination)

-------------------------methods----------------------
ArrowWriter.__enter__(self)
ArrowWriter.__exit__(self, exc_type, exc_val, exc_tb)
ArrowWriter.__init__(self, schema: Optional[pa.Schema]  =  None, features: Optional[Features]  =  None, path: Optional[str]  =  None, stream: Optional[pa.NativeFile]  =  None, fingerprint: Optional[str]  =  None, writer_batch_size: Optional[int]  =  None, hash_salt: Optional[str]  =  None, check_duplicates: Optional[bool]  =  False, disable_nullable: bool  =  False, update_features: bool  =  False, with_metadata: bool  =  True, unit: str  =  "examples", embed_local_files: bool  =  False, storage_options: Optional[dict]  =  None, )
ArrowWriter.__len__(self)
ArrowWriter._build_metadata(info: DatasetInfo, fingerprint: Optional[str]  =  None)
ArrowWriter._build_writer(self, inferred_schema: pa.Schema)
ArrowWriter.check_duplicate_keys(self)
ArrowWriter.close(self)
ArrowWriter.finalize(self, close_stream = True)
ArrowWriter.schema(self)
ArrowWriter.write(self, example: Dict[str, Any], key: Optional[Union[str, int, bytes]]  =  None, writer_batch_size: Optional[int]  =  None, )
ArrowWriter.write_batch(self, batch_examples: Dict[str, List], writer_batch_size: Optional[int]  =  None, )
ArrowWriter.write_examples_on_file(self)
ArrowWriter.write_row(self, row: pa.Table, writer_batch_size: Optional[int]  =  None)
ArrowWriter.write_rows_on_file(self)
ArrowWriter.write_table(self, pa_table: pa.Table, writer_batch_size: Optional[int]  =  None)
BeamWriter.__init__(self, features: Optional[Features]  =  None, schema: Optional[pa.Schema]  =  None, path: Optional[str]  =  None, namespace: Optional[str]  =  None, cache_dir: Optional[str]  =  None, )
BeamWriter.finalize(self, metrics_query_result: dict)
BeamWriter.write_from_pcollection(self, pcoll_examples)
OptimizedTypedSequence.__init__(self, data, type: Optional[FeatureType]  =  None, try_type: Optional[FeatureType]  =  None, col: Optional[str]  =  None, optimized_int_type: Optional[FeatureType]  =  None, )
TypedSequence.__arrow_array__(self, type: Optional[pa.DataType]  =  None)
TypedSequence.__init__(self, data: Iterable, type: Optional[FeatureType]  =  None, try_type: Optional[FeatureType]  =  None, optimized_int_type: Optional[FeatureType]  =  None, )
TypedSequence._infer_custom_type_and_encode(data: Iterable)
TypedSequence.get_inferred_type(self)


repos/datasets/src/datasets/builder.py
-------------------------methods----------------------
ArrowBasedBuilder._generate_tables(self, **kwargs)
ArrowBasedBuilder._get_examples_iterable_for_split(self, split_generator: SplitGenerator)
ArrowBasedBuilder._prepare_split(self, split_generator: SplitGenerator, file_format: str  =  "arrow", num_proc: Optional[int]  =  None, max_shard_size: Optional[Union[str, int]]  =  None, )
ArrowBasedBuilder._prepare_split_single(self, gen_kwargs: dict, fpath: str, file_format: str, max_shard_size: int, job_id: int)
BeamBasedBuilder.__init__(self, *args, beam_runner = None, beam_options = None, **kwargs)
BeamBasedBuilder._build_pcollection(self, pipeline, **kwargs)
BeamBasedBuilder._download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs)
BeamBasedBuilder._generate_examples_from_hf_gcs(self, split: SplitInfo)
BeamBasedBuilder._get_examples_iterable_for_split(self, split: SplitInfo)
BeamBasedBuilder._make_split_generators_kwargs(self, prepare_split_kwargs)
BeamBasedBuilder._prepare_split(self, split_generator, pipeline, file_format = "arrow", max_shard_size: Optional[Union[str, int]]  =  None)
BeamBasedBuilder._remote_cache_dir_from_hf_gcs(self)
BeamBasedBuilder._request_info_from_hf_gcs(self)
BeamBasedBuilder._save_info(self)
BeamBasedBuilder.as_streaming_dataset(self, split: Optional[str]  =  None, )
BuilderConfig.__eq__(self, o)
BuilderConfig.__post_init__(self)
BuilderConfig._resolve_data_files(self, base_path: str, download_config: DownloadConfig)
BuilderConfig.create_config_id(self, config_kwargs: dict, custom_features: Optional[Features]  =  None, )
DatasetBuilder.__getstate__(self)
DatasetBuilder.__init__(self, cache_dir: Optional[str]  =  None, dataset_name: Optional[str]  =  None, config_name: Optional[str]  =  None, hash: Optional[str]  =  None, base_path: Optional[str]  =  None, info: Optional[DatasetInfo]  =  None, features: Optional[Features]  =  None, token: Optional[Union[bool, str]]  =  None, use_auth_token = "deprecated", repo_id: Optional[str]  =  None, data_files: Optional[Union[str, list, dict, DataFilesDict]]  =  None, data_dir: Optional[str]  =  None, storage_options: Optional[dict]  =  None, writer_batch_size: Optional[int]  =  None, name = "deprecated", **config_kwargs, )
DatasetBuilder.__setstate__(self, d)
DatasetBuilder._as_dataset(self, split: Union[ReadInstruction, Split]  =  Split.TRAIN, in_memory: bool  =  False)
DatasetBuilder._as_streaming_dataset_single(self, splits_generator, )
DatasetBuilder._build_cache_dir(self)
DatasetBuilder._build_single_dataset(self, split: Union[str, ReadInstruction, Split], run_post_process: bool, verification_mode: VerificationMode, in_memory: bool  =  False, )
DatasetBuilder._check_legacy_cache(self)
DatasetBuilder._check_legacy_cache2(self, dataset_module: "DatasetModule")
DatasetBuilder._check_manual_download(self, dl_manager)
DatasetBuilder._create_builder_config(self, config_name = None, custom_features = None, **config_kwargs)
DatasetBuilder._download_and_prepare(self, dl_manager, verification_mode, **prepare_split_kwargs)
DatasetBuilder._download_post_processing_resources(self, split: str, resource_name: str, dl_manager: DownloadManager)
DatasetBuilder._download_prepared_from_hf_gcs(self, download_config: DownloadConfig)
DatasetBuilder._get_dataset_fingerprint(self, split: Union[ReadInstruction, Split])
DatasetBuilder._get_examples_iterable_for_split(self, split_generator: SplitGenerator)
DatasetBuilder._info(self)
DatasetBuilder._load_info(self)
DatasetBuilder._make_split_generators_kwargs(self, prepare_split_kwargs)
DatasetBuilder._post_process(self, dataset: Dataset, resources_paths: Mapping[str, str])
DatasetBuilder._post_processing_resources(self, split: str)
DatasetBuilder._prepare_split(self, split_generator: SplitGenerator, file_format: str  =  "arrow", max_shard_size: Optional[Union[str, int]]  =  None, num_proc: Optional[int]  =  None, **kwargs, )
DatasetBuilder._relative_data_dir(self, with_version = True, with_hash = True)
DatasetBuilder._rename(self, src: str, dst: str)
DatasetBuilder._save_info(self)
DatasetBuilder._save_infos(self)
DatasetBuilder._split_generators(self, dl_manager: Union[DownloadManager, StreamingDownloadManager])
DatasetBuilder._use_legacy_cache_dir_if_possible(self, dataset_module: "DatasetModule")
DatasetBuilder.as_dataset(self, split: Optional[Split]  =  None, run_post_process = True, verification_mode: Optional[Union[VerificationMode, str]]  =  None, ignore_verifications = "deprecated", in_memory = False, )
DatasetBuilder.as_streaming_dataset(self, split: Optional[str]  =  None, base_path: Optional[str]  =  None, )
DatasetBuilder.builder_configs(cls)
DatasetBuilder.cache_dir(self)
DatasetBuilder.download_and_prepare(self, output_dir: Optional[str]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, verification_mode: Optional[Union[VerificationMode, str]]  =  None, ignore_verifications = "deprecated", try_from_hf_gcs = "deprecated", dl_manager: Optional[DownloadManager]  =  None, base_path: Optional[str]  =  None, use_auth_token = "deprecated", file_format: str  =  "arrow", max_shard_size: Optional[Union[int, str]]  =  None, num_proc: Optional[int]  =  None, storage_options: Optional[dict]  =  None, **download_and_prepare_kwargs, )
DatasetBuilder.download_post_processing_resources(self, dl_manager)
DatasetBuilder.get_all_exported_dataset_infos(cls)
DatasetBuilder.get_exported_dataset_info(self)
DatasetBuilder.get_imported_module_dir(cls)
DatasetBuilder.manual_download_instructions(self)
GeneratorBasedBuilder._download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs)
GeneratorBasedBuilder._generate_examples(self, **kwargs)
GeneratorBasedBuilder._get_examples_iterable_for_split(self, split_generator: SplitGenerator)
GeneratorBasedBuilder._prepare_split(self, split_generator: SplitGenerator, check_duplicate_keys: bool, file_format = "arrow", num_proc: Optional[int]  =  None, max_shard_size: Optional[Union[int, str]]  =  None, )
GeneratorBasedBuilder._prepare_split_single(self, gen_kwargs: dict, fpath: str, file_format: str, max_shard_size: int, split_info: SplitInfo, check_duplicate_keys: bool, job_id: int, )


repos/datasets/src/datasets/combine.py
-------------------------functions----------------------
concatenate_datasets(dsets: List[DatasetType], info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, axis: int  =  0, )
interleave_datasets(datasets: List[DatasetType], probabilities: Optional[List[float]]  =  None, seed: Optional[int]  =  None, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, stopping_strategy: Literal["first_exhausted", "all_exhausted"]  =  "first_exhausted", )



repos/datasets/src/datasets/commands/__init__.py
-------------------------methods----------------------
BaseDatasetsCLICommand.register_subcommand(parser: ArgumentParser)
BaseDatasetsCLICommand.run(self)


repos/datasets/src/datasets/commands/convert.py


repos/datasets/src/datasets/commands/convert_to_parquet.py
-------------------------functions----------------------
_command_factory(args)

-------------------------methods----------------------
ConvertToParquetCommand.__init__(self, dataset_id: str, token: Optional[str], revision: Optional[str], trust_remote_code: bool, )
ConvertToParquetCommand.register_subcommand(parser)
ConvertToParquetCommand.run(self)


repos/datasets/src/datasets/commands/datasets_cli.py
-------------------------functions----------------------
main()
parse_unknown_args(unknown_args)



repos/datasets/src/datasets/commands/delete_from_hub.py
-------------------------functions----------------------
_command_factory(args)

-------------------------methods----------------------
DeleteFromHubCommand.__init__(self, dataset_id: str, config_name: str, token: Optional[str], revision: Optional[str], )
DeleteFromHubCommand.register_subcommand(parser)
DeleteFromHubCommand.run(self)


repos/datasets/src/datasets/commands/dummy_data.py
-------------------------functions----------------------
dummy_data_command_factory(args)

-------------------------methods----------------------
DummyDataCommand.__init__(self, path_to_dataset: str, auto_generate: bool, n_lines: int, json_field: Optional[str], xml_tag: Optional[str], match_text_files: Optional[str], keep_uncompressed: bool, cache_dir: Optional[str], encoding: Optional[str], )
DummyDataCommand._autogenerate_dummy_data(self, dataset_builder, mock_dl_manager, keep_uncompressed)
DummyDataCommand._print_dummy_data_instructions(self, dataset_builder, mock_dl_manager)
DummyDataCommand.register_subcommand(parser: ArgumentParser)
DummyDataCommand.run(self)
DummyDataGeneratorDownloadManager.__init__(self, mock_download_manager, *args, **kwargs)
DummyDataGeneratorDownloadManager._create_dummy_data(self, src_path: str, dst_path: str, n_lines: int, json_field: Optional[str]  =  None, xml_tag: Optional[str]  =  None, match_text_files: Optional[str]  =  None, encoding: Optional[str]  =  None, )
DummyDataGeneratorDownloadManager._create_xml_dummy_data(src_path, dst_path, xml_tag, n_lines = 5, encoding = DEFAULT_ENCODING)
DummyDataGeneratorDownloadManager.auto_generate_dummy_data_folder(self, n_lines: int  =  5, json_field: Optional[str]  =  None, xml_tag: Optional[str]  =  None, match_text_files: Optional[str]  =  None, encoding: Optional[str]  =  None, )
DummyDataGeneratorDownloadManager.compress_autogenerated_dummy_data(self, path_to_dataset)
DummyDataGeneratorDownloadManager.download(self, url_or_urls)
DummyDataGeneratorDownloadManager.download_and_extract(self, url_or_urls)


repos/datasets/src/datasets/commands/env.py
-------------------------functions----------------------
info_command_factory(_)

-------------------------methods----------------------
EnvironmentCommand.format_dict(d)
EnvironmentCommand.register_subcommand(parser: ArgumentParser)
EnvironmentCommand.run(self)


repos/datasets/src/datasets/commands/run_beam.py
-------------------------functions----------------------
run_beam_command_factory(args, **kwargs)

-------------------------methods----------------------
RunBeamCommand.__init__(self, dataset: str, name: str, cache_dir: str, beam_pipeline_options: str, data_dir: str, all_configs: bool, save_infos: bool, ignore_verifications: bool, force_redownload: bool, **config_kwargs, )
RunBeamCommand.register_subcommand(parser: ArgumentParser)
RunBeamCommand.run(self)


repos/datasets/src/datasets/commands/test.py
-------------------------functions----------------------
_test_command_factory(args)

-------------------------methods----------------------
TestCommand.__init__(self, dataset: str, name: str, cache_dir: str, data_dir: str, all_configs: bool, save_infos: bool, ignore_verifications: bool, force_redownload: bool, clear_cache: bool, num_proc: int, )
TestCommand.register_subcommand(parser: ArgumentParser)
TestCommand.run(self)


repos/datasets/src/datasets/config.py


repos/datasets/src/datasets/data_files.py
-------------------------functions----------------------
_get_data_files_patterns(pattern_resolver: Callable[[str], List[str]])
_get_metadata_files_patterns(pattern_resolver: Callable[[str], List[str]])
_get_origin_metadata(data_files: List[str], download_config: Optional[DownloadConfig]  =  None, max_workers: Optional[int]  =  None, )
_get_single_origin_metadata(data_file: str, download_config: Optional[DownloadConfig]  =  None, )
_is_inside_unrequested_special_dir(matched_rel_path: str, pattern: str)
_is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(matched_rel_path: str, pattern: str)
contains_wildcards(pattern: str)
get_data_patterns(base_path: str, download_config: Optional[DownloadConfig]  =  None)
get_metadata_patterns(base_path: str, download_config: Optional[DownloadConfig]  =  None, )
resolve_pattern(pattern: str, base_path: str, allowed_extensions: Optional[List[str]]  =  None, download_config: Optional[DownloadConfig]  =  None, )
sanitize_patterns(patterns: Union[Dict, List, str])

-------------------------methods----------------------
DataFilesDict.filter_extensions(self, extensions: List[str])
DataFilesDict.from_hf_repo(cls, patterns: Dict[str, Union[List[str], DataFilesList]], dataset_info: huggingface_hub.hf_api.DatasetInfo, base_path: Optional[str]  =  None, allowed_extensions: Optional[List[str]]  =  None, download_config: Optional[DownloadConfig]  =  None, )
DataFilesDict.from_local_or_remote(cls, patterns: Dict[str, Union[List[str], DataFilesList]], base_path: Optional[str]  =  None, allowed_extensions: Optional[List[str]]  =  None, download_config: Optional[DownloadConfig]  =  None, )
DataFilesDict.from_patterns(cls, patterns: Dict[str, Union[List[str], DataFilesList]], base_path: Optional[str]  =  None, allowed_extensions: Optional[List[str]]  =  None, download_config: Optional[DownloadConfig]  =  None, )
DataFilesList.__add__(self, other)
DataFilesList.__init__(self, data_files: List[str], origin_metadata: List[Tuple[str]])
DataFilesList.filter_extensions(self, extensions: List[str])
DataFilesList.from_hf_repo(cls, patterns: List[str], dataset_info: huggingface_hub.hf_api.DatasetInfo, base_path: Optional[str]  =  None, allowed_extensions: Optional[List[str]]  =  None, download_config: Optional[DownloadConfig]  =  None, )
DataFilesList.from_local_or_remote(cls, patterns: List[str], base_path: Optional[str]  =  None, allowed_extensions: Optional[List[str]]  =  None, download_config: Optional[DownloadConfig]  =  None, )
DataFilesList.from_patterns(cls, patterns: List[str], base_path: Optional[str]  =  None, allowed_extensions: Optional[List[str]]  =  None, download_config: Optional[DownloadConfig]  =  None, )
DataFilesPatternsDict.filter_extensions(self, extensions: List[str])
DataFilesPatternsDict.from_patterns(cls, patterns: Dict[str, List[str]], allowed_extensions: Optional[List[str]]  =  None)
DataFilesPatternsDict.resolve(self, base_path: str, download_config: Optional[DownloadConfig]  =  None, )
DataFilesPatternsList.__add__(self, other)
DataFilesPatternsList.__init__(self, patterns: List[str], allowed_extensions: List[Optional[List[str]]], )
DataFilesPatternsList.filter_extensions(self, extensions: List[str])
DataFilesPatternsList.from_patterns(cls, patterns: List[str], allowed_extensions: Optional[List[str]]  =  None)
DataFilesPatternsList.resolve(self, base_path: str, download_config: Optional[DownloadConfig]  =  None, )


repos/datasets/src/datasets/dataset_dict.py
-------------------------methods----------------------
DatasetDict.__enter__(self)
DatasetDict.__exit__(self, exc_type, exc_val, exc_tb)
DatasetDict.__getitem__(self, k)
DatasetDict.__repr__(self)
DatasetDict._check_values_features(self)
DatasetDict._check_values_type(self)
DatasetDict.align_labels_with_mapping(self, label2id: Dict, label_column: str)
DatasetDict.cache_files(self)
DatasetDict.cast(self, features: Features)
DatasetDict.cast_column(self, column: str, feature)
DatasetDict.class_encode_column(self, column: str, include_nulls: bool  =  False)
DatasetDict.cleanup_cache_files(self)
DatasetDict.column_names(self)
DatasetDict.data(self)
DatasetDict.filter(self, function: Optional[Callable]  =  None, with_indices: bool  =  False, with_rank: bool  =  False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, cache_file_names: Optional[Dict[str, Optional[str]]]  =  None, writer_batch_size: Optional[int]  =  1000, fn_kwargs: Optional[dict]  =  None, num_proc: Optional[int]  =  None, desc: Optional[str]  =  None, )
DatasetDict.flatten(self, max_depth = 16)
DatasetDict.flatten_indices(self, keep_in_memory: bool  =  False, cache_file_names: Optional[Dict[str, Optional[str]]]  =  None, writer_batch_size: Optional[int]  =  1000, features: Optional[Features]  =  None, disable_nullable: bool  =  False, num_proc: Optional[int]  =  None, new_fingerprint: Optional[str]  =  None, )
DatasetDict.formatted_as(self, type: Optional[str]  =  None, columns: Optional[List]  =  None, output_all_columns: bool  =  False, **format_kwargs, )
DatasetDict.from_csv(path_or_paths: Dict[str, PathLike], features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, **kwargs, )
DatasetDict.from_json(path_or_paths: Dict[str, PathLike], features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, **kwargs, )
DatasetDict.from_parquet(path_or_paths: Dict[str, PathLike], features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, columns: Optional[List[str]]  =  None, **kwargs, )
DatasetDict.from_text(path_or_paths: Dict[str, PathLike], features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, **kwargs, )
DatasetDict.load_from_disk(dataset_dict_path: PathLike, fs = "deprecated", keep_in_memory: Optional[bool]  =  None, storage_options: Optional[dict]  =  None, )
DatasetDict.map(self, function: Optional[Callable]  =  None, with_indices: bool  =  False, with_rank: bool  =  False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, drop_last_batch: bool  =  False, remove_columns: Optional[Union[str, List[str]]]  =  None, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, cache_file_names: Optional[Dict[str, Optional[str]]]  =  None, writer_batch_size: Optional[int]  =  1000, features: Optional[Features]  =  None, disable_nullable: bool  =  False, fn_kwargs: Optional[dict]  =  None, num_proc: Optional[int]  =  None, desc: Optional[str]  =  None, )
DatasetDict.num_columns(self)
DatasetDict.num_rows(self)
DatasetDict.prepare_for_task(self, task: Union[str, TaskTemplate], id: int  =  0)
DatasetDict.push_to_hub(self, repo_id, config_name: str  =  "default", set_default: Optional[bool]  =  None, data_dir: Optional[str]  =  None, commit_message: Optional[str]  =  None, commit_description: Optional[str]  =  None, private: Optional[bool]  =  False, token: Optional[str]  =  None, revision: Optional[str]  =  None, branch = "deprecated", create_pr: Optional[bool]  =  False, max_shard_size: Optional[Union[int, str]]  =  None, num_shards: Optional[Dict[str, int]]  =  None, embed_external_files: bool  =  True, )
DatasetDict.remove_columns(self, column_names: Union[str, List[str]])
DatasetDict.rename_column(self, original_column_name: str, new_column_name: str)
DatasetDict.rename_columns(self, column_mapping: Dict[str, str])
DatasetDict.reset_format(self)
DatasetDict.save_to_disk(self, dataset_dict_path: PathLike, fs = "deprecated", max_shard_size: Optional[Union[str, int]]  =  None, num_shards: Optional[Dict[str, int]]  =  None, num_proc: Optional[int]  =  None, storage_options: Optional[dict]  =  None, )
DatasetDict.select_columns(self, column_names: Union[str, List[str]])
DatasetDict.set_format(self, type: Optional[str]  =  None, columns: Optional[List]  =  None, output_all_columns: bool  =  False, **format_kwargs, )
DatasetDict.set_transform(self, transform: Optional[Callable], columns: Optional[List]  =  None, output_all_columns: bool  =  False, )
DatasetDict.shape(self)
DatasetDict.shuffle(self, seeds: Optional[Union[int, Dict[str, Optional[int]]]]  =  None, seed: Optional[int]  =  None, generators: Optional[Dict[str, np.random.Generator]]  =  None, keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, indices_cache_file_names: Optional[Dict[str, Optional[str]]]  =  None, writer_batch_size: Optional[int]  =  1000, )
DatasetDict.sort(self, column_names: Union[str, Sequence[str]], reverse: Union[bool, Sequence[bool]]  =  False, kind = "deprecated", null_placement: str  =  "at_end", keep_in_memory: bool  =  False, load_from_cache_file: Optional[bool]  =  None, indices_cache_file_names: Optional[Dict[str, Optional[str]]]  =  None, writer_batch_size: Optional[int]  =  1000, )
DatasetDict.unique(self, column: str)
DatasetDict.with_format(self, type: Optional[str]  =  None, columns: Optional[List]  =  None, output_all_columns: bool  =  False, **format_kwargs, )
DatasetDict.with_transform(self, transform: Optional[Callable], columns: Optional[List]  =  None, output_all_columns: bool  =  False, )
IterableDatasetDict.__repr__(self)
IterableDatasetDict.cast(self, features: Features, )
IterableDatasetDict.cast_column(self, column: str, feature: FeatureType)
IterableDatasetDict.filter(self, function: Optional[Callable]  =  None, with_indices = False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, fn_kwargs: Optional[dict]  =  None, )
IterableDatasetDict.map(self, function: Optional[Callable]  =  None, with_indices: bool  =  False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: int  =  1000, drop_last_batch: bool  =  False, remove_columns: Optional[Union[str, List[str]]]  =  None, fn_kwargs: Optional[dict]  =  None, )
IterableDatasetDict.remove_columns(self, column_names: Union[str, List[str]])
IterableDatasetDict.rename_column(self, original_column_name: str, new_column_name: str)
IterableDatasetDict.rename_columns(self, column_mapping: Dict[str, str])
IterableDatasetDict.select_columns(self, column_names: Union[str, List[str]])
IterableDatasetDict.shuffle(self, seed = None, generator: Optional[np.random.Generator]  =  None, buffer_size: int  =  1000)
IterableDatasetDict.with_format(self, type: Optional[str]  =  None, )


repos/datasets/src/datasets/distributed.py
-------------------------functions----------------------
split_dataset_by_node(dataset: DatasetType, rank: int, world_size: int)



repos/datasets/src/datasets/download/__init__.py


repos/datasets/src/datasets/download/download_config.py
-------------------------methods----------------------
DownloadConfig.__post_init__(self, use_auth_token)
DownloadConfig.__setattr__(self, name, value)
DownloadConfig.copy(self)


repos/datasets/src/datasets/download/download_manager.py
-------------------------methods----------------------
DownloadManager.__init__(self, dataset_name: Optional[str]  =  None, data_dir: Optional[str]  =  None, download_config: Optional[DownloadConfig]  =  None, base_path: Optional[str]  =  None, record_checksums = True, )
DownloadManager._download_batched(self, url_or_filenames: List[str], download_config: DownloadConfig, )
DownloadManager._download_single(self, url_or_filename: str, download_config: DownloadConfig)
DownloadManager._record_sizes_checksums(self, url_or_urls: NestedDataStructure, downloaded_path_or_paths: NestedDataStructure)
DownloadManager.delete_extracted_files(self)
DownloadManager.download(self, url_or_urls)
DownloadManager.download_and_extract(self, url_or_urls)
DownloadManager.download_custom(self, url_or_urls, custom_download)
DownloadManager.downloaded_size(self)
DownloadManager.extract(self, path_or_paths, num_proc = "deprecated")
DownloadManager.get_recorded_sizes_checksums(self)
DownloadManager.iter_archive(self, path_or_buf: Union[str, io.BufferedReader])
DownloadManager.iter_files(self, paths: Union[str, List[str]])
DownloadManager.manage_extracted_files(self)
DownloadManager.manual_dir(self)
DownloadManager.ship_files_with_pipeline(downloaded_path_or_paths, pipeline)
GenerateMode.help_message(self)


repos/datasets/src/datasets/download/mock_download_manager.py
-------------------------methods----------------------
MockDownloadManager.__init__(self, dataset_name: str, config: str, version: Union[Version, str], cache_dir: Optional[str]  =  None, use_local_dummy_data: bool  =  False, load_existing_dummy_data: bool  =  True, download_callbacks: Optional[List[Callable]]  =  None, )
MockDownloadManager.create_dummy_data_dict(self, path_to_dummy_data, data_url)
MockDownloadManager.create_dummy_data_list(self, path_to_dummy_data, data_url)
MockDownloadManager.create_dummy_data_single(self, path_to_dummy_data, data_url)
MockDownloadManager.delete_extracted_files(self)
MockDownloadManager.download(self, data_url, *args)
MockDownloadManager.download_and_extract(self, data_url, *args)
MockDownloadManager.download_custom(self, data_url, custom_download)
MockDownloadManager.download_dummy_data(self)
MockDownloadManager.dummy_data_folder(self)
MockDownloadManager.dummy_file(self)
MockDownloadManager.dummy_zip_file(self)
MockDownloadManager.extract(self, path, *args, **kwargs)
MockDownloadManager.get_recorded_sizes_checksums(self)
MockDownloadManager.github_path_to_dummy_data(self)
MockDownloadManager.iter_archive(self, path)
MockDownloadManager.iter_files(self, paths)
MockDownloadManager.local_path_to_dummy_data(self)
MockDownloadManager.manage_extracted_files(self)
MockDownloadManager.manual_dir(self)


repos/datasets/src/datasets/download/streaming_download_manager.py
-------------------------methods----------------------
StreamingDownloadManager.__init__(self, dataset_name: Optional[str]  =  None, data_dir: Optional[str]  =  None, download_config: Optional[DownloadConfig]  =  None, base_path: Optional[str]  =  None, )
StreamingDownloadManager._download_single(self, urlpath: str)
StreamingDownloadManager._extract(self, urlpath: str)
StreamingDownloadManager.download(self, url_or_urls)
StreamingDownloadManager.download_and_extract(self, url_or_urls)
StreamingDownloadManager.extract(self, url_or_urls)
StreamingDownloadManager.iter_archive(self, urlpath_or_buf: Union[str, io.BufferedReader])
StreamingDownloadManager.iter_files(self, urlpaths: Union[str, List[str]])
StreamingDownloadManager.manual_dir(self)


repos/datasets/src/datasets/exceptions.py
-------------------------methods----------------------
DatasetGenerationCastError.from_cast_error(cls, cast_error: CastError, builder_name: str, gen_kwargs: Dict[str, Any], token: Optional[Union[bool, str]], )


repos/datasets/src/datasets/features/__init__.py


repos/datasets/src/datasets/features/audio.py
-------------------------methods----------------------
Audio.__call__(self)
Audio.cast_storage(self, storage: Union[pa.StringArray, pa.StructArray])
Audio.decode_example(self, value: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]  =  None)
Audio.embed_storage(self, storage: pa.StructArray)
Audio.encode_example(self, value: Union[str, bytes, dict])
Audio.flatten(self)


repos/datasets/src/datasets/features/features.py
-------------------------functions----------------------
_align_features(features_list: List[Features])
_arrow_to_datasets_dtype(arrow_type: pa.DataType)
_cast_to_python_objects(obj: Any, only_1d_for_numpy: bool, optimize_list_casting: bool)
_check_if_features_can_be_aligned(features_list: List[Features])
_check_non_null_non_empty_recursive(obj, schema: Optional[FeatureType]  =  None)
_is_zero_copy_only(pa_type: pa.DataType, unnest: bool  =  False)
_visit(feature: FeatureType, func: Callable[[FeatureType], Optional[FeatureType]])
any_np_array_to_pyarrow_listarray(data: Union[np.ndarray, List], type: pa.DataType  =  None)
cast_to_python_objects(obj: Any, only_1d_for_numpy = False, optimize_list_casting = True)
contains_any_np_array(data: Any)
decode_nested_example(schema, obj, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]  =  None)
encode_nested_example(schema, obj, level = 0)
generate_from_arrow_type(pa_type: pa.DataType)
generate_from_dict(obj: Any)
get_nested_type(schema: FeatureType)
keep_features_dicts_synced(func)
list_of_np_array_to_pyarrow_listarray(l_arr: List[np.ndarray], type: pa.DataType  =  None)
list_of_pa_arrays_to_pyarrow_listarray(l_arr: List[Optional[pa.Array]])
numpy_to_pyarrow_listarray(arr: np.ndarray, type: pa.DataType  =  None)
pandas_types_mapper(dtype)
register_feature(feature_cls: type, feature_type: str, )
require_decoding(feature: FeatureType, ignore_decode_attribute: bool  =  False)
require_storage_cast(feature: FeatureType)
require_storage_embed(feature: FeatureType)
string_to_arrow(datasets_dtype: str)
to_pyarrow_listarray(data: Any, pa_type: _ArrayXDExtensionType)

-------------------------methods----------------------
ArrayExtensionArray.__array__(self)
ArrayExtensionArray.__getitem__(self, i)
ArrayExtensionArray.to_numpy(self, zero_copy_only = True)
ArrayExtensionArray.to_pylist(self)
ClassLabel.__call__(self)
ClassLabel.__post_init__(self, num_classes, names_file)
ClassLabel._load_names_from_file(names_filepath)
ClassLabel._strval2int(self, value: str)
ClassLabel.cast_storage(self, storage: Union[pa.StringArray, pa.IntegerArray])
ClassLabel.encode_example(self, example_data)
ClassLabel.int2str(self, values: Union[int, Iterable])
ClassLabel.str2int(self, values: Union[str, Iterable])
Features.__init__(*args, **kwargs)
Features.__reduce__(self)
Features._from_yaml_list(cls, yaml_data: list)
Features._to_yaml_list(self)
Features.arrow_schema(self)
Features.copy(self)
Features.decode_batch(self, batch: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]  =  None)
Features.decode_column(self, column: list, column_name: str)
Features.decode_example(self, example: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]  =  None)
Features.encode_batch(self, batch)
Features.encode_column(self, column, column_name: str)
Features.encode_example(self, example)
Features.flatten(self, max_depth = 16)
Features.from_arrow_schema(cls, pa_schema: pa.Schema)
Features.from_dict(cls, dic)
Features.reorder_fields_as(self, other: "Features")
Features.to_dict(self)
Features.type(self)
PandasArrayExtensionArray.__array__(self, dtype = None)
PandasArrayExtensionArray.__eq__(self, other)
PandasArrayExtensionArray.__getitem__(self, item: Union[int, slice, np.ndarray])
PandasArrayExtensionArray.__init__(self, data: np.ndarray, copy: bool  =  False)
PandasArrayExtensionArray.__len__(self)
PandasArrayExtensionArray.__setitem__(self, key: Union[int, slice, np.ndarray], value: Any)
PandasArrayExtensionArray._concat_same_type(cls, to_concat: Sequence_["PandasArrayExtensionArray"])
PandasArrayExtensionArray._from_sequence(cls, scalars, dtype: Optional[PandasArrayExtensionDtype]  =  None, copy: bool  =  False)
PandasArrayExtensionArray.copy(self, deep: bool  =  False)
PandasArrayExtensionArray.dtype(self)
PandasArrayExtensionArray.isna(self)
PandasArrayExtensionArray.nbytes(self)
PandasArrayExtensionArray.take(self, indices: Sequence_[int], allow_fill: bool  =  False, fill_value: bool  =  None)
PandasArrayExtensionDtype.__from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray])
PandasArrayExtensionDtype.__init__(self, value_type: Union["PandasArrayExtensionDtype", np.dtype])
PandasArrayExtensionDtype.construct_array_type(cls)
PandasArrayExtensionDtype.kind(self)
PandasArrayExtensionDtype.name(self)
PandasArrayExtensionDtype.type(self)
PandasArrayExtensionDtype.value_type(self)
Value.__call__(self)
Value.__post_init__(self)
Value.encode_example(self, value)
_ArrayXD.__call__(self)
_ArrayXD.__post_init__(self)
_ArrayXD.encode_example(self, value)
_ArrayXDExtensionType.__arrow_ext_class__(self)
_ArrayXDExtensionType.__arrow_ext_deserialize__(cls, storage_type, serialized)
_ArrayXDExtensionType.__arrow_ext_serialize__(self)
_ArrayXDExtensionType.__hash__(self)
_ArrayXDExtensionType.__init__(self, shape: tuple, dtype: str)
_ArrayXDExtensionType.__reduce__(self)
_ArrayXDExtensionType._generate_dtype(self, dtype)
_ArrayXDExtensionType.to_pandas_dtype(self)


repos/datasets/src/datasets/features/image.py
-------------------------functions----------------------
encode_np_array(array: np.ndarray)
encode_pil_image(image: "PIL.Image.Image")
image_to_bytes(image: "PIL.Image.Image")
list_image_compression_formats()
objects_to_list_of_image_dicts(objs: Union[List[str], List[dict], List[np.ndarray], List["PIL.Image.Image"]], )

-------------------------methods----------------------
Image.__call__(self)
Image.cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray])
Image.decode_example(self, value: dict, token_per_repo_id = None)
Image.embed_storage(self, storage: pa.StructArray)
Image.encode_example(self, value: Union[str, bytes, dict, np.ndarray, "PIL.Image.Image"])
Image.flatten(self)


repos/datasets/src/datasets/features/translation.py
-------------------------methods----------------------
Translation.__call__(self)
Translation.flatten(self)
TranslationVariableLanguages.__call__(self)
TranslationVariableLanguages.__post_init__(self)
TranslationVariableLanguages.encode_example(self, translation_dict)
TranslationVariableLanguages.flatten(self)


repos/datasets/src/datasets/filesystems/__init__.py
-------------------------functions----------------------
extract_path_from_uri(dataset_path: str)
is_remote_filesystem(fs: fsspec.AbstractFileSystem)
rename(fs: fsspec.AbstractFileSystem, src: str, dst: str)



repos/datasets/src/datasets/filesystems/compression.py
-------------------------methods----------------------
BaseCompressedFileFileSystem.__init__(self, fo: str  =  "", target_protocol: Optional[str]  =  None, target_options: Optional[dict]  =  None, **kwargs)
BaseCompressedFileFileSystem._get_dirs(self)
BaseCompressedFileFileSystem._open(self, path: str, mode: str  =  "rb", block_size = None, autocommit = True, cache_options = None, **kwargs, )
BaseCompressedFileFileSystem._strip_protocol(cls, path)
BaseCompressedFileFileSystem.cat(self, path: str)


repos/datasets/src/datasets/filesystems/s3filesystem.py


repos/datasets/src/datasets/fingerprint.py
-------------------------functions----------------------
disable_caching()
enable_caching()
fingerprint_transform(inplace: bool, use_kwargs: Optional[List[str]]  =  None, ignore_kwargs: Optional[List[str]]  =  None, fingerprint_names: Optional[List[str]]  =  None, randomized_function: bool  =  False, version: Optional[str]  =  None, )
format_kwargs_for_fingerprint(func: Callable, args: Tuple, kwargs: Dict[str, Any], use_kwargs: Optional[List[str]]  =  None, ignore_kwargs: Optional[List[str]]  =  None, randomized_function: bool  =  False, )
format_transform_for_fingerprint(func: Callable, version: Optional[str]  =  None)
generate_fingerprint(dataset: "Dataset")
generate_random_fingerprint(nbits: int  =  64)
get_datasets_with_cache_file_in_temp_dir()
get_temporary_cache_files_directory()
hashregister(*types)
is_caching_enabled()
maybe_register_dataset_for_temp_dir_deletion(dataset)
set_caching_enabled(boolean: bool)
update_fingerprint(fingerprint, transform, transform_args)
validate_fingerprint(fingerprint: str, max_length = 64)

-------------------------methods----------------------
Hasher.__init__(self)
Hasher.hash(cls, value: Any)
Hasher.hash_bytes(cls, value: Union[bytes, List[bytes]])
Hasher.hash_default(cls, value: Any)
Hasher.hexdigest(self)
Hasher.update(self, value: Any)
_TempCacheDir.__init__(self)
_TempCacheDir._cleanup(self)
_TempCacheDir.cleanup(self)


repos/datasets/src/datasets/formatting/__init__.py
-------------------------functions----------------------
_register_formatter(formatter_cls: type, format_type: Optional[str], aliases: Optional[List[str]]  =  None, )
_register_unavailable_formatter(unavailable_error: Exception, format_type: Optional[str], aliases: Optional[List[str]]  =  None)
get_format_type_from_alias(format_type: Optional[str])
get_formatter(format_type: Optional[str], **format_kwargs)



repos/datasets/src/datasets/formatting/formatting.py
-------------------------functions----------------------
_check_valid_column_key(key: str, columns: List[str])
_check_valid_index_key(key: Union[int, slice, range, Iterable], size: int)
_is_array_with_nulls(pa_array: pa.Array)
_is_range_contiguous(key: range)
_query_table(table: Table, key: Union[int, slice, range, str, Iterable])
_query_table_with_indices_mapping(table: Table, key: Union[int, slice, range, str, Iterable], indices: Table)
_raise_bad_key_type(key: Any)
_unnest(py_dict: Dict[str, List[T]])
format_table(table: Table, key: Union[int, slice, range, str, Iterable], formatter: Formatter, format_columns: Optional[list]  =  None, output_all_columns = False, )
key_to_query_type(key: Union[int, slice, range, str, Iterable])
query_table(table: Table, key: Union[int, slice, range, str, Iterable], indices: Optional[Table]  =  None, )

-------------------------methods----------------------
ArrowFormatter.format_batch(self, pa_table: pa.Table)
ArrowFormatter.format_column(self, pa_table: pa.Table)
ArrowFormatter.format_row(self, pa_table: pa.Table)
BaseArrowExtractor.extract_batch(self, pa_table: pa.Table)
BaseArrowExtractor.extract_column(self, pa_table: pa.Table)
BaseArrowExtractor.extract_row(self, pa_table: pa.Table)
CustomFormatter.__init__(self, transform: Callable[[dict], dict], features = None, **kwargs)
CustomFormatter.format_batch(self, pa_table: pa.Table)
CustomFormatter.format_column(self, pa_table: pa.Table)
CustomFormatter.format_row(self, pa_table: pa.Table)
Formatter.__call__(self, pa_table: pa.Table, query_type: str)
Formatter.__init__(self, features: Optional[Features]  =  None)
Formatter.format_batch(self, pa_table: pa.Table)
Formatter.format_column(self, pa_table: pa.Table)
Formatter.format_row(self, pa_table: pa.Table)
LazyBatch.format(self, key)
LazyDict.__contains__(self, key)
LazyDict.__copy__(self)
LazyDict.__delitem__(self, key)
LazyDict.__getitem__(self, key)
LazyDict.__init__(self, pa_table: pa.Table, formatter: "Formatter")
LazyDict.__iter__(self)
LazyDict.__len__(self)
LazyDict.__repr__(self)
LazyDict.__setitem__(self, key, value)
LazyDict._format_all(self)
LazyDict.copy(self)
LazyDict.format(self, key)
LazyDict.fromkeys(cls, iterable, value = None)
LazyRow.format(self, key)
NumpyArrowExtractor.__init__(self, **np_array_kwargs)
NumpyArrowExtractor._arrow_array_to_numpy(self, pa_array: pa.Array)
NumpyArrowExtractor.extract_batch(self, pa_table: pa.Table)
NumpyArrowExtractor.extract_column(self, pa_table: pa.Table)
NumpyArrowExtractor.extract_row(self, pa_table: pa.Table)
PandasArrowExtractor.extract_batch(self, pa_table: pa.Table)
PandasArrowExtractor.extract_column(self, pa_table: pa.Table)
PandasArrowExtractor.extract_row(self, pa_table: pa.Table)
PandasFeaturesDecoder.__init__(self, features: Optional[Features])
PandasFeaturesDecoder.decode_batch(self, batch: pd.DataFrame)
PandasFeaturesDecoder.decode_column(self, column: pd.Series, column_name: str)
PandasFeaturesDecoder.decode_row(self, row: pd.DataFrame)
PandasFormatter.format_batch(self, pa_table: pa.Table)
PandasFormatter.format_column(self, pa_table: pa.Table)
PandasFormatter.format_row(self, pa_table: pa.Table)
PythonArrowExtractor.extract_batch(self, pa_table: pa.Table)
PythonArrowExtractor.extract_column(self, pa_table: pa.Table)
PythonArrowExtractor.extract_row(self, pa_table: pa.Table)
PythonFeaturesDecoder.__init__(self, features: Optional[Features])
PythonFeaturesDecoder.decode_batch(self, batch: dict)
PythonFeaturesDecoder.decode_column(self, column: list, column_name: str)
PythonFeaturesDecoder.decode_row(self, row: dict)
PythonFormatter.__init__(self, features = None, lazy = False)
PythonFormatter.format_batch(self, pa_table: pa.Table)
PythonFormatter.format_column(self, pa_table: pa.Table)
PythonFormatter.format_row(self, pa_table: pa.Table)
SimpleArrowExtractor.extract_batch(self, pa_table: pa.Table)
SimpleArrowExtractor.extract_column(self, pa_table: pa.Table)
SimpleArrowExtractor.extract_row(self, pa_table: pa.Table)
TensorFormatter.recursive_tensorize(self, data_struct: dict)


repos/datasets/src/datasets/formatting/jax_formatter.py
-------------------------methods----------------------
JaxFormatter.__init__(self, features = None, device = None, **jnp_array_kwargs)
JaxFormatter._consolidate(self, column)
JaxFormatter._map_devices_to_str()
JaxFormatter._recursive_tensorize(self, data_struct)
JaxFormatter._tensorize(self, value)
JaxFormatter.format_batch(self, pa_table: pa.Table)
JaxFormatter.format_column(self, pa_table: pa.Table)
JaxFormatter.format_row(self, pa_table: pa.Table)
JaxFormatter.recursive_tensorize(self, data_struct: dict)


repos/datasets/src/datasets/formatting/np_formatter.py
-------------------------methods----------------------
NumpyFormatter.__init__(self, features = None, **np_array_kwargs)
NumpyFormatter._consolidate(self, column)
NumpyFormatter._recursive_tensorize(self, data_struct)
NumpyFormatter._tensorize(self, value)
NumpyFormatter.format_batch(self, pa_table: pa.Table)
NumpyFormatter.format_column(self, pa_table: pa.Table)
NumpyFormatter.format_row(self, pa_table: pa.Table)
NumpyFormatter.recursive_tensorize(self, data_struct: dict)


repos/datasets/src/datasets/formatting/polars_formatter.py
-------------------------methods----------------------
PolarsArrowExtractor.extract_batch(self, pa_table: pa.Table)
PolarsArrowExtractor.extract_column(self, pa_table: pa.Table)
PolarsArrowExtractor.extract_row(self, pa_table: pa.Table)
PolarsFeaturesDecoder.__init__(self, features: Optional[Features])
PolarsFeaturesDecoder.decode_batch(self, batch: "pl.DataFrame")
PolarsFeaturesDecoder.decode_column(self, column: "pl.Series", column_name: str)
PolarsFeaturesDecoder.decode_row(self, row: "pl.DataFrame")
PolarsFormatter.__init__(self, features = None, **np_array_kwargs)
PolarsFormatter.format_batch(self, pa_table: pa.Table)
PolarsFormatter.format_column(self, pa_table: pa.Table)
PolarsFormatter.format_row(self, pa_table: pa.Table)


repos/datasets/src/datasets/formatting/tf_formatter.py
-------------------------methods----------------------
TFFormatter.__init__(self, features = None, **tf_tensor_kwargs)
TFFormatter._consolidate(self, column)
TFFormatter._recursive_tensorize(self, data_struct)
TFFormatter._tensorize(self, value)
TFFormatter.format_batch(self, pa_table: pa.Table)
TFFormatter.format_column(self, pa_table: pa.Table)
TFFormatter.format_row(self, pa_table: pa.Table)
TFFormatter.recursive_tensorize(self, data_struct: dict)


repos/datasets/src/datasets/formatting/torch_formatter.py
-------------------------methods----------------------
TorchFormatter.__init__(self, features = None, **torch_tensor_kwargs)
TorchFormatter._consolidate(self, column)
TorchFormatter._recursive_tensorize(self, data_struct)
TorchFormatter._tensorize(self, value)
TorchFormatter.format_batch(self, pa_table: pa.Table)
TorchFormatter.format_column(self, pa_table: pa.Table)
TorchFormatter.format_row(self, pa_table: pa.Table)
TorchFormatter.recursive_tensorize(self, data_struct: dict)


repos/datasets/src/datasets/hub.py
-------------------------functions----------------------
_delete_files(dataset_id, revision = None, token = None)
convert_to_parquet(repo_id: str, revision: Optional[str]  =  None, token: Optional[Union[bool, str]]  =  None, trust_remote_code: Optional[bool]  =  None, )
delete_from_hub(repo_id: str, config_name: str, revision: Optional[str]  =  None, token: Optional[Union[bool, str]]  =  None, )



repos/datasets/src/datasets/info.py
-------------------------methods----------------------
DatasetInfo.__post_init__(self)
DatasetInfo._dump_info(self, file, pretty_print = False)
DatasetInfo._dump_license(self, file)
DatasetInfo._from_yaml_dict(cls, yaml_data: dict)
DatasetInfo._to_yaml_dict(self)
DatasetInfo.copy(self)
DatasetInfo.from_dict(cls, dataset_info_dict: dict)
DatasetInfo.from_directory(cls, dataset_info_dir: str, fs = "deprecated", storage_options: Optional[dict]  =  None)
DatasetInfo.from_merge(cls, dataset_infos: List["DatasetInfo"])
DatasetInfo.update(self, other_dataset_info: "DatasetInfo", ignore_none = True)
DatasetInfo.write_to_directory(self, dataset_info_dir, pretty_print = False, fs = "deprecated", storage_options: Optional[dict]  =  None)
DatasetInfosDict.from_dataset_card_data(cls, dataset_card_data: DatasetCardData)
DatasetInfosDict.from_directory(cls, dataset_infos_dir)
DatasetInfosDict.to_dataset_card_data(self, dataset_card_data: DatasetCardData)
DatasetInfosDict.write_to_directory(self, dataset_infos_dir, overwrite = False, pretty_print = False)
MetricInfo.__post_init__(self)
MetricInfo.from_dict(cls, metric_info_dict: dict)
MetricInfo.from_directory(cls, metric_info_dir)
MetricInfo.write_to_directory(self, metric_info_dir, pretty_print = False)
PostProcessedInfo.__post_init__(self)
PostProcessedInfo.from_dict(cls, post_processed_info_dict: dict)


repos/datasets/src/datasets/inspect.py
-------------------------functions----------------------
get_dataset_config_info(path: str, config_name: Optional[str]  =  None, data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, revision: Optional[Union[str, Version]]  =  None, token: Optional[Union[bool, str]]  =  None, use_auth_token = "deprecated", **config_kwargs, )
get_dataset_config_names(path: str, revision: Optional[Union[str, Version]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, data_files: Optional[Union[Dict, List, str]]  =  None, **download_kwargs, )
get_dataset_default_config_name(path: str, revision: Optional[Union[str, Version]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, data_files: Optional[Union[Dict, List, str]]  =  None, **download_kwargs, )
get_dataset_split_names(path: str, config_name: Optional[str]  =  None, data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, revision: Optional[Union[str, Version]]  =  None, token: Optional[Union[bool, str]]  =  None, use_auth_token = "deprecated", **config_kwargs, )
inspect_dataset(path: str, local_path: str, download_config: Optional[DownloadConfig]  =  None, **download_kwargs)
inspect_metric(path: str, local_path: str, download_config: Optional[DownloadConfig]  =  None, **download_kwargs)
list_datasets(with_community_datasets = True, with_details = False)
list_metrics(with_community_metrics = True, with_details = False)



repos/datasets/src/datasets/io/__init__.py


repos/datasets/src/datasets/io/abc.py
-------------------------methods----------------------
AbstractDatasetInputStream.__init__(self, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, streaming: bool  =  False, num_proc: Optional[int]  =  None, **kwargs, )
AbstractDatasetInputStream.read(self)
AbstractDatasetReader.__init__(self, path_or_paths: Optional[NestedDataStructureLike[PathLike]]  =  None, split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, streaming: bool  =  False, num_proc: Optional[int]  =  None, **kwargs, )
AbstractDatasetReader.read(self)


repos/datasets/src/datasets/io/csv.py
-------------------------methods----------------------
CsvDatasetReader.__init__(self, path_or_paths: NestedDataStructureLike[PathLike], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, streaming: bool  =  False, num_proc: Optional[int]  =  None, **kwargs, )
CsvDatasetReader.read(self)
CsvDatasetWriter.__init__(self, dataset: Dataset, path_or_buf: Union[PathLike, BinaryIO], batch_size: Optional[int]  =  None, num_proc: Optional[int]  =  None, storage_options: Optional[dict]  =  None, **to_csv_kwargs, )
CsvDatasetWriter._batch_csv(self, args)
CsvDatasetWriter._write(self, file_obj: BinaryIO, header, index, **to_csv_kwargs)
CsvDatasetWriter.write(self)


repos/datasets/src/datasets/io/generator.py
-------------------------methods----------------------
GeneratorDatasetInputStream.__init__(self, generator: Callable, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, streaming: bool  =  False, gen_kwargs: Optional[dict]  =  None, num_proc: Optional[int]  =  None, **kwargs, )
GeneratorDatasetInputStream.read(self)


repos/datasets/src/datasets/io/json.py
-------------------------methods----------------------
JsonDatasetReader.__init__(self, path_or_paths: NestedDataStructureLike[PathLike], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, streaming: bool  =  False, field: Optional[str]  =  None, num_proc: Optional[int]  =  None, **kwargs, )
JsonDatasetReader.read(self)
JsonDatasetWriter.__init__(self, dataset: Dataset, path_or_buf: Union[PathLike, BinaryIO], batch_size: Optional[int]  =  None, num_proc: Optional[int]  =  None, storage_options: Optional[dict]  =  None, **to_json_kwargs, )
JsonDatasetWriter._batch_json(self, args)
JsonDatasetWriter._write(self, file_obj: BinaryIO, orient, lines, **to_json_kwargs, )
JsonDatasetWriter.write(self)


repos/datasets/src/datasets/io/parquet.py
-------------------------functions----------------------
get_writer_batch_size(features: Features)

-------------------------methods----------------------
ParquetDatasetReader.__init__(self, path_or_paths: NestedDataStructureLike[PathLike], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, streaming: bool  =  False, num_proc: Optional[int]  =  None, **kwargs, )
ParquetDatasetReader.read(self)
ParquetDatasetWriter.__init__(self, dataset: Dataset, path_or_buf: Union[PathLike, BinaryIO], batch_size: Optional[int]  =  None, storage_options: Optional[dict]  =  None, **parquet_writer_kwargs, )
ParquetDatasetWriter._write(self, file_obj: BinaryIO, batch_size: int, **parquet_writer_kwargs)
ParquetDatasetWriter.write(self)


repos/datasets/src/datasets/io/spark.py
-------------------------methods----------------------
SparkDatasetReader.__init__(self, df: pyspark.sql.DataFrame, split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, streaming: bool  =  True, cache_dir: str  =  None, keep_in_memory: bool  =  False, working_dir: str  =  None, load_from_cache_file: bool  =  True, file_format: str  =  "arrow", **kwargs, )
SparkDatasetReader.read(self)


repos/datasets/src/datasets/io/sql.py
-------------------------methods----------------------
SqlDatasetReader.__init__(self, sql: Union[str, "sqlalchemy.sql.Selectable"], con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"], features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, **kwargs, )
SqlDatasetReader.read(self)
SqlDatasetWriter.__init__(self, dataset: Dataset, name: str, con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"], batch_size: Optional[int]  =  None, num_proc: Optional[int]  =  None, **to_sql_kwargs, )
SqlDatasetWriter._batch_sql(self, args)
SqlDatasetWriter._write(self, index, **to_sql_kwargs)
SqlDatasetWriter.write(self)


repos/datasets/src/datasets/io/text.py
-------------------------methods----------------------
TextDatasetReader.__init__(self, path_or_paths: NestedDataStructureLike[PathLike], split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, cache_dir: str  =  None, keep_in_memory: bool  =  False, streaming: bool  =  False, num_proc: Optional[int]  =  None, **kwargs, )
TextDatasetReader.read(self)


repos/datasets/src/datasets/iterable_dataset.py
-------------------------functions----------------------
_apply_feature_types_on_batch(batch: dict, features: Features, token_per_repo_id: Dict[str, Union[str, bool, None]])
_apply_feature_types_on_example(example: dict, features: Features, token_per_repo_id: Dict[str, Union[str, bool, None]])
_batch_arrow_tables(iterable: Iterable[Tuple[Key, pa.Table]], batch_size: Optional[int], drop_last_batch: bool  =  False, )
_batch_to_examples(batch: Dict[str, list])
_check_column_names(column_names: List[str])
_concatenate_iterable_datasets(dsets: List[IterableDataset], info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, axis: int  =  0, )
_convert_to_arrow(iterable: Iterable[Tuple[Key, dict]], batch_size: int, drop_last_batch: bool  =  False, )
_examples_to_batch(examples: List[Dict[str, Any]])
_infer_features_from_batch(batch: Dict[str, list], try_features: Optional[Features]  =  None)
_interleave_iterable_datasets(datasets: List[IterableDataset], probabilities: Optional[List[float]]  =  None, seed: Optional[int]  =  None, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, stopping_strategy: Literal["first_exhausted", "all_exhausted"]  =  "first_exhausted", )
_maybe_add_torch_iterable_dataset_parent_class(cls)
_rename_columns_fn(example: Dict, column_mapping: Dict[str, str])
_split_by_node_iterable_dataset(dataset: IterableDataset, rank: int, world_size: int)
add_column_fn(example: Dict, idx: int, name: str, column: List[Dict])
identity_func(x)

-------------------------methods----------------------
ArrowExamplesIterable.__init__(self, generate_tables_fn: Callable[..., Tuple[Key, pa.Table]], kwargs: dict)
ArrowExamplesIterable.__iter__(self)
ArrowExamplesIterable._iter_arrow(self)
ArrowExamplesIterable.n_shards(self)
ArrowExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
ArrowExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
BufferShuffledExamplesIterable.__init__(self, ex_iterable: _BaseExamplesIterable, buffer_size: int, generator: np.random.Generator)
BufferShuffledExamplesIterable.__iter__(self)
BufferShuffledExamplesIterable._iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size = 1000)
BufferShuffledExamplesIterable.n_shards(self)
BufferShuffledExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
BufferShuffledExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
CyclingMultiSourcesExamplesIterable.__init__(self, ex_iterables: List[_BaseExamplesIterable], stopping_strategy: Literal["first_exhausted", "all_exhausted"]  =  "first_exhausted", )
CyclingMultiSourcesExamplesIterable.__iter__(self)
CyclingMultiSourcesExamplesIterable._get_indices_iterator(self)
CyclingMultiSourcesExamplesIterable.n_shards(self)
CyclingMultiSourcesExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
CyclingMultiSourcesExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
ExamplesIterable.__init__(self, generate_examples_fn: Callable[..., Tuple[Key, dict]], kwargs: dict)
ExamplesIterable.__iter__(self)
ExamplesIterable.n_shards(self)
ExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
ExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
FilteredExamplesIterable.__init__(self, ex_iterable: _BaseExamplesIterable, function: Callable, with_indices: bool  =  False, input_columns: Optional[List[str]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, fn_kwargs: Optional[dict]  =  None, formatting: Optional["FormattingConfig"]  =  None, format_type = "deprecated", )
FilteredExamplesIterable.__iter__(self)
FilteredExamplesIterable._iter(self)
FilteredExamplesIterable._iter_arrow(self)
FilteredExamplesIterable.n_shards(self)
FilteredExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
FilteredExamplesIterable.shuffle_data_sources(self, seed: Optional[int])
FormattingConfig.__post_init__(self)
HorizontallyConcatenatedMultiSourcesExamplesIterable.__init__(self, ex_iterables: List[_BaseExamplesIterable])
HorizontallyConcatenatedMultiSourcesExamplesIterable.__iter__(self)
HorizontallyConcatenatedMultiSourcesExamplesIterable.n_shards(self)
HorizontallyConcatenatedMultiSourcesExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
HorizontallyConcatenatedMultiSourcesExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
IterableDataset.__getstate__(self)
IterableDataset.__init__(self, ex_iterable: _BaseExamplesIterable, info: Optional[DatasetInfo]  =  None, split: Optional[NamedSplit]  =  None, formatting: Optional[FormattingConfig]  =  None, shuffling: Optional[ShufflingConfig]  =  None, distributed: Optional[DistributedConfig]  =  None, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]  =  None, format_type = "deprecated", )
IterableDataset.__iter__(self)
IterableDataset.__repr__(self)
IterableDataset.__setstate__(self, d)
IterableDataset._effective_generator(self)
IterableDataset._head(self, n = 5)
IterableDataset._is_main_process(self)
IterableDataset._iter_pytorch(self)
IterableDataset._prepare_ex_iterable_for_iteration(self)
IterableDataset._resolve_features(self)
IterableDataset._step(self, step: int, offset: int)
IterableDataset.add_column(self, name: str, column: Union[list, np.array])
IterableDataset.cast(self, features: Features, )
IterableDataset.cast_column(self, column: str, feature: FeatureType)
IterableDataset.column_names(self)
IterableDataset.filter(self, function: Optional[Callable]  =  None, with_indices = False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, fn_kwargs: Optional[dict]  =  None, )
IterableDataset.from_file(filename: str)
IterableDataset.from_generator(generator: Callable, features: Optional[Features]  =  None, gen_kwargs: Optional[dict]  =  None, )
IterableDataset.from_spark(df: "pyspark.sql.DataFrame", split: Optional[NamedSplit]  =  None, features: Optional[Features]  =  None, **kwargs, )
IterableDataset.iter(self, batch_size: int, drop_last_batch: bool  =  False)
IterableDataset.map(self, function: Optional[Callable]  =  None, with_indices: bool  =  False, input_columns: Optional[Union[str, List[str]]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, drop_last_batch: bool  =  False, remove_columns: Optional[Union[str, List[str]]]  =  None, features: Optional[Features]  =  None, fn_kwargs: Optional[dict]  =  None, )
IterableDataset.n_shards(self)
IterableDataset.remove_columns(self, column_names: Union[str, List[str]])
IterableDataset.rename_column(self, original_column_name: str, new_column_name: str)
IterableDataset.rename_columns(self, column_mapping: Dict[str, str])
IterableDataset.select_columns(self, column_names: Union[str, List[str]])
IterableDataset.set_epoch(self, epoch: int)
IterableDataset.shuffle(self, seed = None, generator: Optional[np.random.Generator]  =  None, buffer_size: int  =  1000)
IterableDataset.skip(self, n: int)
IterableDataset.take(self, n: int)
IterableDataset.with_format(self, type: Optional[str]  =  None, )
MappedExamplesIterable.__init__(self, ex_iterable: _BaseExamplesIterable, function: Callable, with_indices: bool  =  False, input_columns: Optional[List[str]]  =  None, batched: bool  =  False, batch_size: Optional[int]  =  1000, drop_last_batch: bool  =  False, remove_columns: Optional[List[str]]  =  None, fn_kwargs: Optional[dict]  =  None, formatting: Optional["FormattingConfig"]  =  None, format_type = "deprecated", )
MappedExamplesIterable.__iter__(self)
MappedExamplesIterable._iter(self)
MappedExamplesIterable._iter_arrow(self)
MappedExamplesIterable.n_shards(self)
MappedExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
MappedExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
RandomlyCyclingMultiSourcesExamplesIterable.__init__(self, ex_iterables: List[_BaseExamplesIterable], generator: np.random.Generator, probabilities: Optional[List[float]]  =  None, stopping_strategy: Literal["first_exhausted", "all_exhausted"]  =  "first_exhausted", )
RandomlyCyclingMultiSourcesExamplesIterable._get_indices_iterator(self)
RandomlyCyclingMultiSourcesExamplesIterable._iter_random_indices(rng: np.random.Generator, num_sources: int, random_batch_size = 1000, p: Optional[List[float]]  =  None, )
RandomlyCyclingMultiSourcesExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
RandomlyCyclingMultiSourcesExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
SelectColumnsIterable.__init__(self, ex_iterable: _BaseExamplesIterable, column_names: List[str])
SelectColumnsIterable.__iter__(self)
SelectColumnsIterable._iter_arrow(self)
SelectColumnsIterable.n_shards(self)
SelectColumnsIterable.shard_data_sources(self, worker_id: int, num_workers: int)
SelectColumnsIterable.shuffle_data_sources(self, generator: np.random.Generator)
ShuffledDataSourcesArrowExamplesIterable.__init__(self, generate_tables_fn: Callable[..., Tuple[Key, pa.Table]], kwargs: dict, generator: np.random.Generator, )
ShuffledDataSourcesArrowExamplesIterable.__iter__(self)
ShuffledDataSourcesArrowExamplesIterable._iter_arrow(self)
ShuffledDataSourcesArrowExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
ShuffledDataSourcesExamplesIterable.__init__(self, generate_examples_fn: Callable[..., Tuple[Key, dict]], kwargs: dict, generator: np.random.Generator)
ShuffledDataSourcesExamplesIterable.__iter__(self)
ShuffledDataSourcesExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
SkipExamplesIterable.__init__(self, ex_iterable: _BaseExamplesIterable, n: int)
SkipExamplesIterable.__iter__(self)
SkipExamplesIterable.n_shards(self)
SkipExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
StepExamplesIterable.__init__(self, ex_iterable: _BaseExamplesIterable, step: int, offset: int)
StepExamplesIterable.__iter__(self)
StepExamplesIterable.n_shards(self)
StepExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
StepExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
TakeExamplesIterable.__init__(self, ex_iterable: _BaseExamplesIterable, n: int)
TakeExamplesIterable.__iter__(self)
TakeExamplesIterable.n_shards(self)
TakeExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
TakeExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
TakeExamplesIterable.split_number(num, n)
TypedExamplesIterable.__init__(self, ex_iterable: _BaseExamplesIterable, features: Features, token_per_repo_id: Dict[str, Union[str, bool, None]], )
TypedExamplesIterable.__iter__(self)
TypedExamplesIterable._iter_arrow(self)
TypedExamplesIterable.n_shards(self)
TypedExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
TypedExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
VerticallyConcatenatedMultiSourcesExamplesIterable.__init__(self, ex_iterables: List[_BaseExamplesIterable])
VerticallyConcatenatedMultiSourcesExamplesIterable.__iter__(self)
VerticallyConcatenatedMultiSourcesExamplesIterable._iter_arrow(self)
VerticallyConcatenatedMultiSourcesExamplesIterable.n_shards(self)
VerticallyConcatenatedMultiSourcesExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
VerticallyConcatenatedMultiSourcesExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
_BaseExamplesIterable.__init__(self)
_BaseExamplesIterable.__iter__(self)
_BaseExamplesIterable.n_shards(self)
_BaseExamplesIterable.shard_data_sources(self, worker_id: int, num_workers: int)
_BaseExamplesIterable.shuffle_data_sources(self, generator: np.random.Generator)
_BaseExamplesIterable.split_shard_indices_by_worker(self, worker_id: int, num_workers: int)
_HasNextIterator.__init__(self, it)
_HasNextIterator.__iter__(self)
_HasNextIterator.__next__(self)
_HasNextIterator.hasnext(self)


repos/datasets/src/datasets/keyhash.py
-------------------------functions----------------------
_as_bytes(hash_data: Union[str, int, bytes])

-------------------------methods----------------------
DuplicatedKeysError.__init__(self, key, duplicate_key_indices, fix_msg = "")
InvalidKeyError.__init__(self, hash_data)
KeyHasher.__init__(self, hash_salt: str)
KeyHasher.hash(self, key: Union[str, int, bytes])


repos/datasets/src/datasets/load.py
-------------------------functions----------------------
_copy_script_and_other_resources_in_importable_dir(name: str, importable_directory_path: str, subdirectory_name: str, original_local_path: str, local_imports: List[Tuple[str, str]], additional_files: List[Tuple[str, str]], download_mode: Optional[Union[DownloadMode, str]], )
_create_importable_file(local_path: str, local_imports: List[Tuple[str, str]], additional_files: List[Tuple[str, str]], dynamic_modules_path: str, module_namespace: str, subdirectory_name: str, name: str, download_mode: DownloadMode, )
_download_additional_modules(name: str, base_path: str, imports: Tuple[str, str, str, str], download_config: Optional[DownloadConfig])
_get_importable_file_path(dynamic_modules_path: str, module_namespace: str, subdirectory_name: str, name: str, )
_load_importable_file(dynamic_modules_path: str, module_namespace: str, subdirectory_name: str, name: str, )
_raise_timeout_error(signum, frame)
configure_builder_class(builder_cls: Type[DatasetBuilder], builder_configs: List[BuilderConfig], default_config_name: Optional[str], dataset_name: str, )
create_builder_configs_from_metadata_configs(module_path: str, metadata_configs: MetadataConfigs, supports_metadata: bool, base_path: Optional[str]  =  None, default_builder_kwargs: Dict[str, Any]  =  None, download_config: Optional[DownloadConfig]  =  None, )
dataset_module_factory(path: str, revision: Optional[Union[str, Version]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, data_dir: Optional[str]  =  None, data_files: Optional[Union[Dict, List, str, DataFilesDict]]  =  None, cache_dir: Optional[str]  =  None, trust_remote_code: Optional[bool]  =  None, _require_default_config_name = True, _require_custom_configs = False, **download_kwargs, )
files_to_hash(file_paths: List[str])
get_dataset_builder_class(dataset_module: "DatasetModule", dataset_name: Optional[str]  =  None)
import_main_class(module_path, dataset = True)
increase_load_count(name: str, resource_type: str)
infer_module_for_data_files(data_files: DataFilesDict, path: Optional[str]  =  None, download_config: Optional[DownloadConfig]  =  None)
infer_module_for_data_files_list(data_files_list: DataFilesList, download_config: Optional[DownloadConfig]  =  None)
infer_module_for_data_files_list_in_archives(data_files_list: DataFilesList, download_config: Optional[DownloadConfig]  =  None)
init_dynamic_modules(name: str  =  config.MODULE_NAME_FOR_DYNAMIC_MODULES, hf_modules_cache: Optional[Union[Path, str]]  =  None)
load_dataset(path: str, name: Optional[str]  =  None, data_dir: Optional[str]  =  None, data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]  =  None, split: Optional[Union[str, Split]]  =  None, cache_dir: Optional[str]  =  None, features: Optional[Features]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, verification_mode: Optional[Union[VerificationMode, str]]  =  None, ignore_verifications = "deprecated", keep_in_memory: Optional[bool]  =  None, save_infos: bool  =  False, revision: Optional[Union[str, Version]]  =  None, token: Optional[Union[bool, str]]  =  None, use_auth_token = "deprecated", task = "deprecated", streaming: bool  =  False, num_proc: Optional[int]  =  None, storage_options: Optional[Dict]  =  None, trust_remote_code: bool  =  None, **config_kwargs, )
load_dataset_builder(path: str, name: Optional[str]  =  None, data_dir: Optional[str]  =  None, data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]  =  None, cache_dir: Optional[str]  =  None, features: Optional[Features]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, revision: Optional[Union[str, Version]]  =  None, token: Optional[Union[bool, str]]  =  None, use_auth_token = "deprecated", storage_options: Optional[Dict]  =  None, trust_remote_code: Optional[bool]  =  None, _require_default_config_name = True, **config_kwargs, )
load_from_disk(dataset_path: str, fs = "deprecated", keep_in_memory: Optional[bool]  =  None, storage_options: Optional[dict]  =  None)
load_metric(path: str, config_name: Optional[str]  =  None, process_id: int  =  0, num_process: int  =  1, cache_dir: Optional[str]  =  None, experiment_id: Optional[str]  =  None, keep_in_memory: bool  =  False, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, revision: Optional[Union[str, Version]]  =  None, trust_remote_code: Optional[bool]  =  None, **metric_init_kwargs, )
metric_module_factory(path: str, revision: Optional[Union[str, Version]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, trust_remote_code: Optional[bool]  =  None, **download_kwargs, )
resolve_trust_remote_code(trust_remote_code: Optional[bool], repo_id: str)

-------------------------methods----------------------
CachedDatasetModuleFactory.__init__(self, name: str, cache_dir: Optional[str]  =  None, dynamic_modules_path: Optional[str]  =  None, )
CachedDatasetModuleFactory.get_module(self)
CachedMetricModuleFactory.__init__(self, name: str, dynamic_modules_path: Optional[str]  =  None, )
CachedMetricModuleFactory.get_module(self)
GithubMetricModuleFactory.__init__(self, name: str, revision: Optional[Union[str, Version]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, trust_remote_code: Optional[str]  =  None, )
GithubMetricModuleFactory.download_loading_script(self, revision: Optional[str])
GithubMetricModuleFactory.get_module(self)
HubDatasetModuleFactoryWithParquetExport.__init__(self, name: str, revision: Optional[str]  =  None, download_config: Optional[DownloadConfig]  =  None, )
HubDatasetModuleFactoryWithParquetExport.get_module(self)
HubDatasetModuleFactoryWithScript.__init__(self, name: str, revision: Optional[Union[str, Version]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, trust_remote_code: Optional[bool]  =  None, )
HubDatasetModuleFactoryWithScript.download_dataset_infos_file(self)
HubDatasetModuleFactoryWithScript.download_dataset_readme_file(self)
HubDatasetModuleFactoryWithScript.download_loading_script(self)
HubDatasetModuleFactoryWithScript.get_module(self)
HubDatasetModuleFactoryWithoutScript.__init__(self, name: str, revision: Optional[Union[str, Version]]  =  None, data_dir: Optional[str]  =  None, data_files: Optional[Union[str, List, Dict]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, )
HubDatasetModuleFactoryWithoutScript.get_module(self)
LocalDatasetModuleFactoryWithScript.__init__(self, path: str, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, trust_remote_code: Optional[bool]  =  None, )
LocalDatasetModuleFactoryWithScript.get_module(self)
LocalDatasetModuleFactoryWithoutScript.__init__(self, path: str, data_dir: Optional[str]  =  None, data_files: Optional[Union[str, List, Dict]]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, )
LocalDatasetModuleFactoryWithoutScript.get_module(self)
LocalMetricModuleFactory.__init__(self, path: str, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, dynamic_modules_path: Optional[str]  =  None, trust_remote_code: Optional[str]  =  None, )
LocalMetricModuleFactory.get_module(self)
PackagedDatasetModuleFactory.__init__(self, name: str, data_dir: Optional[str]  =  None, data_files: Optional[Union[str, List, Dict]]  =  None, download_config: Optional[DownloadConfig]  =  None, download_mode: Optional[Union[DownloadMode, str]]  =  None, )
PackagedDatasetModuleFactory.get_module(self)
_DatasetModuleFactory.get_module(self)
_InitializeConfiguredDatasetBuilder.__call__(self, builder_cls, metadata_configs, default_config_name, name)
_MetricModuleFactory.get_module(self)


repos/datasets/src/datasets/metric.py
-------------------------functions----------------------
summarize_if_long_list(obj)

-------------------------methods----------------------
FileFreeLock.__init__(self, lock_file, *args, **kwargs)
FileFreeLock._acquire(self)
FileFreeLock._release(self)
Metric.__del__(self)
Metric.__init__(self, config_name: Optional[str]  =  None, keep_in_memory: bool  =  False, cache_dir: Optional[str]  =  None, num_process: int  =  1, process_id: int  =  0, seed: Optional[int]  =  None, experiment_id: Optional[str]  =  None, max_concurrent_cache_files: int  =  10000, timeout: Union[int, float]  =  100, **kwargs, )
Metric.__len__(self)
Metric.__repr__(self)
Metric._build_data_dir(self)
Metric._check_all_processes_locks(self)
Metric._check_rendez_vous(self)
Metric._compute(self, *, predictions = None, references = None, **kwargs)
Metric._create_cache_file(self, timeout = 1)
Metric._download_and_prepare(self, dl_manager)
Metric._finalize(self)
Metric._get_all_cache_files(self)
Metric._info(self)
Metric._init_writer(self, timeout = 1)
Metric.add(self, *, prediction = None, reference = None, **kwargs)
Metric.add_batch(self, *, predictions = None, references = None, **kwargs)
Metric.compute(self, *, predictions = None, references = None, **kwargs)
Metric.download_and_prepare(self, download_config: Optional[DownloadConfig]  =  None, dl_manager: Optional[DownloadManager]  =  None, )
MetricInfoMixin.__init__(self, info: MetricInfo)
MetricInfoMixin.citation(self)
MetricInfoMixin.codebase_urls(self)
MetricInfoMixin.description(self)
MetricInfoMixin.experiment_id(self)
MetricInfoMixin.features(self)
MetricInfoMixin.format(self)
MetricInfoMixin.homepage(self)
MetricInfoMixin.info(self)
MetricInfoMixin.inputs_description(self)
MetricInfoMixin.license(self)
MetricInfoMixin.name(self)
MetricInfoMixin.reference_urls(self)
MetricInfoMixin.streamable(self)


repos/datasets/src/datasets/naming.py
-------------------------functions----------------------
camelcase_to_snakecase(name)
filename_prefix_for_name(name)
filename_prefix_for_split(name, split)
filenames_for_dataset_split(path, dataset_name, split, filetype_suffix = None, shard_lengths = None)
filepattern_for_dataset_split(dataset_name, split, data_dir, filetype_suffix = None)
snakecase_to_camelcase(name)



repos/datasets/src/datasets/packaged_modules/__init__.py
-------------------------functions----------------------
_hash_python_lines(lines: List[str])



repos/datasets/src/datasets/parallel/__init__.py


repos/datasets/src/datasets/parallel/parallel.py
-------------------------functions----------------------
_map_with_joblib(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func)
_map_with_multiprocessing_pool(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func)
parallel_backend(backend_name: str)
parallel_map(function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, single_map_nested_func)



repos/datasets/src/datasets/search.py
-------------------------methods----------------------
BaseIndex.load(cls, file: Union[str, PurePath])
BaseIndex.save(self, file: Union[str, PurePath])
BaseIndex.search(self, query, k: int  =  10, **kwargs)
BaseIndex.search_batch(self, queries, k: int  =  10, **kwargs)
ElasticSearchIndex.__init__(self, host: Optional[str]  =  None, port: Optional[int]  =  None, es_client: Optional["Elasticsearch"]  =  None, es_index_name: Optional[str]  =  None, es_index_config: Optional[dict]  =  None, )
ElasticSearchIndex.add_documents(self, documents: Union[List[str], "Dataset"], column: Optional[str]  =  None)
ElasticSearchIndex.search(self, query: str, k = 10, **kwargs)
ElasticSearchIndex.search_batch(self, queries, k: int  =  10, max_workers = 10, **kwargs)
FaissIndex.__init__(self, device: Optional[Union[int, List[int]]]  =  None, string_factory: Optional[str]  =  None, metric_type: Optional[int]  =  None, custom_index: Optional["faiss.Index"]  =  None, )
FaissIndex._faiss_index_to_device(index: "faiss.Index", device: Optional[Union[int, List[int]]]  =  None)
FaissIndex.add_vectors(self, vectors: Union[np.array, "Dataset"], column: Optional[str]  =  None, batch_size: int  =  1000, train_size: Optional[int]  =  None, faiss_verbose: Optional[bool]  =  None, )
FaissIndex.load(cls, file: Union[str, PurePath], device: Optional[Union[int, List[int]]]  =  None, storage_options: Optional[Dict]  =  None, )
FaissIndex.save(self, file: Union[str, PurePath], storage_options: Optional[Dict]  =  None)
FaissIndex.search(self, query: np.array, k = 10, **kwargs)
FaissIndex.search_batch(self, queries: np.array, k = 10, **kwargs)
IndexableMixin.__getitem__(self, key)
IndexableMixin.__init__(self)
IndexableMixin.__len__(self)
IndexableMixin._check_index_is_initialized(self, index_name: str)
IndexableMixin.add_elasticsearch_index(self, column: str, index_name: Optional[str]  =  None, host: Optional[str]  =  None, port: Optional[int]  =  None, es_client: Optional["Elasticsearch"]  =  None, es_index_name: Optional[str]  =  None, es_index_config: Optional[dict]  =  None, )
IndexableMixin.add_faiss_index(self, column: str, index_name: Optional[str]  =  None, device: Optional[Union[int, List[int]]]  =  None, string_factory: Optional[str]  =  None, metric_type: Optional[int]  =  None, custom_index: Optional["faiss.Index"]  =  None, batch_size: int  =  1000, train_size: Optional[int]  =  None, faiss_verbose: bool  =  False, )
IndexableMixin.add_faiss_index_from_external_arrays(self, external_arrays: np.array, index_name: str, device: Optional[Union[int, List[int]]]  =  None, string_factory: Optional[str]  =  None, metric_type: Optional[int]  =  None, custom_index: Optional["faiss.Index"]  =  None, batch_size: int  =  1000, train_size: Optional[int]  =  None, faiss_verbose: bool  =  False, )
IndexableMixin.drop_index(self, index_name: str)
IndexableMixin.get_index(self, index_name: str)
IndexableMixin.get_nearest_examples(self, index_name: str, query: Union[str, np.array], k: int  =  10, **kwargs)
IndexableMixin.get_nearest_examples_batch(self, index_name: str, queries: Union[List[str], np.array], k: int  =  10, **kwargs)
IndexableMixin.is_index_initialized(self, index_name: str)
IndexableMixin.list_indexes(self)
IndexableMixin.load_elasticsearch_index(self, index_name: str, es_index_name: str, host: Optional[str]  =  None, port: Optional[int]  =  None, es_client: Optional["Elasticsearch"]  =  None, es_index_config: Optional[dict]  =  None, )
IndexableMixin.load_faiss_index(self, index_name: str, file: Union[str, PurePath], device: Optional[Union[int, List[int]]]  =  None, storage_options: Optional[Dict]  =  None, )
IndexableMixin.save_faiss_index(self, index_name: str, file: Union[str, PurePath], storage_options: Optional[Dict]  =  None)
IndexableMixin.search(self, index_name: str, query: Union[str, np.array], k: int  =  10, **kwargs)
IndexableMixin.search_batch(self, index_name: str, queries: Union[List[str], np.array], k: int  =  10, **kwargs)


repos/datasets/src/datasets/splits.py
-------------------------methods----------------------
NamedSplitAll.__init__(self)
NamedSplitAll.__repr__(self)
NamedSplitAll.get_read_instruction(self, split_dict)
PercentSliceMeta.__getitem__(cls, slice_value)
Split.__new__(cls, name)
SplitBase.__add__(self, other)
SplitBase.__eq__(self, other)
SplitBase.__ne__(self, other)
SplitBase.get_read_instruction(self, split_dict)
SplitBase.subsplit(self, arg = None, k = None, percent = None, weighted = None)
SplitDict.__getitem__(self, key: Union[SplitBase, str])
SplitDict.__init__(self, *args, dataset_name = None, **kwargs)
SplitDict.__setitem__(self, key: Union[SplitBase, str], value: SplitInfo)
SplitDict._from_yaml_list(cls, yaml_data: list)
SplitDict._to_yaml_list(self)
SplitDict.add(self, split_info: SplitInfo)
SplitDict.copy(self)
SplitDict.from_split_dict(cls, split_infos: Union[List, Dict], dataset_name: Optional[str]  =  None)
SplitDict.to_split_dict(self)
SplitDict.total_num_examples(self)
SplitGenerator.__post_init__(self)
SplitInfo.file_instructions(self)
SplitReadInstruction.__add__(self, other)
SplitReadInstruction.__getitem__(self, slice_value)
SplitReadInstruction.__init__(self, split_info = None)
SplitReadInstruction.add(self, sliced_split)
SplitReadInstruction.get_list_sliced_split_info(self)
SubSplitInfo.file_instructions(self)
SubSplitInfo.num_examples(self)
_SplitMerged.__init__(self, split1, split2)
_SplitMerged.__repr__(self)
_SplitMerged.get_read_instruction(self, split_dict)
_SubSplit.__init__(self, split, slice_value)
_SubSplit.__repr__(self)
_SubSplit.get_read_instruction(self, split_dict)


repos/datasets/src/datasets/streaming.py
-------------------------functions----------------------
extend_dataset_builder_for_streaming(builder: "DatasetBuilder")
extend_module_for_streaming(module_path, download_config: Optional[DownloadConfig]  =  None)



repos/datasets/src/datasets/table.py
-------------------------functions----------------------
_are_list_values_of_length(array: pa.ListArray, length: int)
_combine_list_array_offsets_with_mask(array: pa.ListArray)
_deepcopy(x, memo: dict)
_in_memory_arrow_table_from_buffer(buffer: pa.Buffer)
_in_memory_arrow_table_from_file(filename: str)
_interpolation_search(arr: List[int], x: int)
_memory_mapped_arrow_table_from_file(filename: str)
_memory_mapped_record_batch_reader_from_file(filename: str)
_short_str(value: Any)
_storage_type(type: pa.DataType)
_wrap_for_chunked_arrays(func)
array_cast(array: pa.Array, pa_type: pa.DataType, allow_primitive_to_str: bool  =  True, allow_decimal_to_str: bool  =  True)
cast_array_to_feature(array: pa.Array, feature: "FeatureType", allow_primitive_to_str: bool  =  True, allow_decimal_to_str: bool  =  True)
cast_table_to_features(table: pa.Table, features: "Features")
cast_table_to_schema(table: pa.Table, schema: pa.Schema)
concat_tables(tables: List[Table], axis: int  =  0)
embed_array_storage(array: pa.Array, feature: "FeatureType")
embed_table_storage(table: pa.Table)
inject_arrow_table_documentation(arrow_table_method)
list_table_cache_files(table: Table)
read_schema_from_file(filename: str)
table_cast(table: pa.Table, schema: pa.Schema)
table_flatten(table: pa.Table)
table_iter(table: Table, batch_size: int, drop_last_batch = False)
table_visitor(table: pa.Table, function: Callable[[pa.Array], None])

-------------------------methods----------------------
CastError.__init__(self, *args, table_column_names: List[str], requested_column_names: List[str])
CastError.__reduce__(self)
CastError.details(self)
ConcatenationTable.__getstate__(self)
ConcatenationTable.__init__(self, table: pa.Table, blocks: List[List[TableBlock]])
ConcatenationTable.__setstate__(self, state)
ConcatenationTable._concat_blocks(blocks: List[Union[TableBlock, pa.Table]], axis: int  =  0)
ConcatenationTable._concat_blocks_horizontally_and_vertically(cls, blocks: List[List[TableBlock]])
ConcatenationTable._consolidate_blocks(cls, blocks: TableBlockContainer)
ConcatenationTable._merge_blocks(cls, blocks: TableBlockContainer, axis: Optional[int]  =  None)
ConcatenationTable._slices(self)
ConcatenationTable.add_column(self, *args, **kwargs)
ConcatenationTable.append_column(self, *args, **kwargs)
ConcatenationTable.cast(self, target_schema, *args, **kwargs)
ConcatenationTable.combine_chunks(self, *args, **kwargs)
ConcatenationTable.drop(self, columns, *args, **kwargs)
ConcatenationTable.filter(self, mask, *args, **kwargs)
ConcatenationTable.flatten(self, *args, **kwargs)
ConcatenationTable.from_blocks(cls, blocks: TableBlockContainer)
ConcatenationTable.from_tables(cls, tables: List[Union[pa.Table, Table]], axis: int  =  0)
ConcatenationTable.remove_column(self, i, *args, **kwargs)
ConcatenationTable.rename_columns(self, names, *args, **kwargs)
ConcatenationTable.replace_schema_metadata(self, *args, **kwargs)
ConcatenationTable.select(self, columns, *args, **kwargs)
ConcatenationTable.set_column(self, *args, **kwargs)
ConcatenationTable.slice(self, offset = 0, length = None)
InMemoryTable.add_column(self, *args, **kwargs)
InMemoryTable.append_column(self, *args, **kwargs)
InMemoryTable.cast(self, *args, **kwargs)
InMemoryTable.combine_chunks(self, *args, **kwargs)
InMemoryTable.drop(self, *args, **kwargs)
InMemoryTable.filter(self, *args, **kwargs)
InMemoryTable.flatten(self, *args, **kwargs)
InMemoryTable.from_arrays(cls, *args, **kwargs)
InMemoryTable.from_batches(cls, *args, **kwargs)
InMemoryTable.from_buffer(cls, buffer: pa.Buffer)
InMemoryTable.from_file(cls, filename: str)
InMemoryTable.from_pandas(cls, *args, **kwargs)
InMemoryTable.from_pydict(cls, *args, **kwargs)
InMemoryTable.from_pylist(cls, mapping, *args, **kwargs)
InMemoryTable.remove_column(self, *args, **kwargs)
InMemoryTable.rename_columns(self, *args, **kwargs)
InMemoryTable.replace_schema_metadata(self, *args, **kwargs)
InMemoryTable.select(self, *args, **kwargs)
InMemoryTable.set_column(self, *args, **kwargs)
InMemoryTable.slice(self, offset = 0, length = None)
IndexedTableMixin.__init__(self, table: pa.Table)
IndexedTableMixin.fast_gather(self, indices: Union[List[int], np.ndarray])
IndexedTableMixin.fast_slice(self, offset = 0, length = None)
MemoryMappedTable.__getstate__(self)
MemoryMappedTable.__init__(self, table: pa.Table, path: str, replays: Optional[List[Replay]]  =  None)
MemoryMappedTable.__setstate__(self, state)
MemoryMappedTable._append_replay(self, replay: Replay)
MemoryMappedTable._apply_replays(table: pa.Table, replays: Optional[List[Replay]]  =  None)
MemoryMappedTable.add_column(self, *args, **kwargs)
MemoryMappedTable.append_column(self, *args, **kwargs)
MemoryMappedTable.cast(self, *args, **kwargs)
MemoryMappedTable.combine_chunks(self, *args, **kwargs)
MemoryMappedTable.drop(self, *args, **kwargs)
MemoryMappedTable.filter(self, *args, **kwargs)
MemoryMappedTable.flatten(self, *args, **kwargs)
MemoryMappedTable.from_file(cls, filename: str, replays = None)
MemoryMappedTable.remove_column(self, *args, **kwargs)
MemoryMappedTable.rename_columns(self, *args, **kwargs)
MemoryMappedTable.replace_schema_metadata(self, *args, **kwargs)
MemoryMappedTable.select(self, *args, **kwargs)
MemoryMappedTable.set_column(self, *args, **kwargs)
MemoryMappedTable.slice(self, offset = 0, length = None)
Table.__deepcopy__(self, memo: dict)
Table.__eq__(self, other)
Table.__getitem__(self, i)
Table.__init__(self, table: pa.Table)
Table.__len__(self)
Table.__repr__(self)
Table.__str__(self)
Table.add_column(self, *args, **kwargs)
Table.append_column(self, *args, **kwargs)
Table.cast(self, *args, **kwargs)
Table.column(self, *args, **kwargs)
Table.column_names(self)
Table.columns(self)
Table.combine_chunks(self, *args, **kwargs)
Table.drop(self, *args, **kwargs)
Table.equals(self, *args, **kwargs)
Table.field(self, *args, **kwargs)
Table.filter(self, *args, **kwargs)
Table.flatten(self, *args, **kwargs)
Table.itercolumns(self, *args, **kwargs)
Table.nbytes(self)
Table.num_columns(self)
Table.num_rows(self)
Table.remove_column(self, *args, **kwargs)
Table.rename_columns(self, *args, **kwargs)
Table.replace_schema_metadata(self, *args, **kwargs)
Table.schema(self)
Table.select(self, *args, **kwargs)
Table.set_column(self, *args, **kwargs)
Table.shape(self)
Table.slice(self, *args, **kwargs)
Table.to_batches(self, *args, **kwargs)
Table.to_pandas(self, *args, **kwargs)
Table.to_pydict(self, *args, **kwargs)
Table.to_pylist(self, *args, **kwargs)
Table.to_reader(self, max_chunksize: Optional[int]  =  None)
Table.to_string(self, *args, **kwargs)
Table.validate(self, *args, **kwargs)


repos/datasets/src/datasets/tasks/__init__.py
-------------------------functions----------------------
task_template_from_dict(task_template_dict: dict)



repos/datasets/src/datasets/tasks/audio_classification.py
-------------------------methods----------------------
AudioClassification.align_with_features(self, features)
AudioClassification.column_mapping(self)


repos/datasets/src/datasets/tasks/automatic_speech_recognition.py
-------------------------methods----------------------
AutomaticSpeechRecognition.align_with_features(self, features)
AutomaticSpeechRecognition.column_mapping(self)


repos/datasets/src/datasets/tasks/base.py
-------------------------methods----------------------
TaskTemplate.align_with_features(self: T, features: Features)
TaskTemplate.column_mapping(self)
TaskTemplate.features(self)
TaskTemplate.from_dict(cls: Type[T], template_dict: dict)


repos/datasets/src/datasets/tasks/image_classification.py
-------------------------methods----------------------
ImageClassification.align_with_features(self, features)
ImageClassification.column_mapping(self)


repos/datasets/src/datasets/tasks/language_modeling.py
-------------------------methods----------------------
LanguageModeling.column_mapping(self)


repos/datasets/src/datasets/tasks/question_answering.py
-------------------------methods----------------------
QuestionAnsweringExtractive.column_mapping(self)


repos/datasets/src/datasets/tasks/summarization.py
-------------------------methods----------------------
Summarization.column_mapping(self)


repos/datasets/src/datasets/tasks/text_classification.py
-------------------------methods----------------------
TextClassification.align_with_features(self, features)
TextClassification.column_mapping(self)


repos/datasets/src/datasets/utils/__init__.py


repos/datasets/src/datasets/utils/_dataset_viewer.py
-------------------------functions----------------------
get_exported_dataset_infos(dataset: str, revision: str, token: Optional[Union[str, bool]])
get_exported_parquet_files(dataset: str, revision: str, token: Optional[Union[str, bool]])



repos/datasets/src/datasets/utils/_dill.py
-------------------------functions----------------------
_save_regexPattern(pickler, obj)
_save_set(pickler, obj)
_save_spacyLanguage(pickler, obj)
_save_tiktokenEncoding(pickler, obj)
_save_torchGenerator(pickler, obj)
_save_torchTensor(pickler, obj)
_save_transformersPreTrainedTokenizerBase(pickler, obj)
dump(obj, file)
dumps(obj)
pklregister(t)

-------------------------methods----------------------
Pickler._batch_setitems(self, items)
Pickler.memoize(self, obj)
Pickler.save(self, obj, save_persistent_id = True)


repos/datasets/src/datasets/utils/_filelock.py
-------------------------methods----------------------
FileLock.__init__(self, lock_file, *args, **kwargs)
FileLock.hash_filename_if_too_long(cls, path: str)


repos/datasets/src/datasets/utils/beam_utils.py
-------------------------functions----------------------
download_remote_to_local(remote_file_path, local_file_path, force_download = False)
upload_local_to_remote(local_file_path, remote_file_path, force_upload = False)

-------------------------methods----------------------
BeamPipeline.is_local(self)


repos/datasets/src/datasets/utils/deprecation_utils.py
-------------------------functions----------------------
deprecated(help_message: Optional[str]  =  None)

-------------------------methods----------------------
DeprecatedEnum.__new__(cls, value)
DeprecatedEnum.deprecate(self)
DeprecatedEnum.help_message(self)
OnAccess.__call__(cls, value, names = None, *, module = None, qualname = None, type = None, start = 1)
OnAccess.__getattribute__(cls, name)
OnAccess.__getitem__(cls, name)


repos/datasets/src/datasets/utils/doc_utils.py
-------------------------functions----------------------
is_documented_by(function_with_docstring: Callable)



repos/datasets/src/datasets/utils/download_manager.py


repos/datasets/src/datasets/utils/experimental.py
-------------------------functions----------------------
experimental(fn: Callable)



repos/datasets/src/datasets/utils/extract.py
-------------------------methods----------------------
BaseExtractor.extract()
BaseExtractor.is_extractable(cls, path: Union[Path, str], **kwargs) -> bool: ...@staticmethod@abstractmethodinput_path: Union[Path, str], output_path: Union[Path, str]) -> None: ...)
Bzip2Extractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
ExtractManager.__init__(self, cache_dir: Optional[str]  =  None)
ExtractManager._do_extract(self, output_path: str, force_extract: bool)
ExtractManager._get_output_path(self, path: str)
ExtractManager.extract(self, input_path: str, force_extract: bool  =  False)
Extractor._get_magic_number_max_length(cls)
Extractor._read_magic_number(path: Union[Path, str], magic_number_length: int)
Extractor.extract(cls, input_path: Union[Path, str], output_path: Union[Path, str], extractor_format: Optional[str]  =  None, # <Added version="2.4.0"/>extractor = "2.4.0"/>extractor: Optional[BaseExtractor] = "deprecated", )
Extractor.infer_extractor_format(cls, path: Union[Path, str]) -> Optional[str]:  # <Added version = "2.4.0"/>)path, magic_number_max_length)))
Extractor.is_extractable(cls, path: Union[Path, str], return_extractor: bool  =  False)
GzipExtractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
Lz4Extractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
MagicNumberBaseExtractor.is_extractable(cls, path: Union[Path, str], magic_number: bytes  =  b"")
MagicNumberBaseExtractor.read_magic_number(path: Union[Path, str], magic_number_length: int)
RarExtractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
SevenZipExtractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
TarExtractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
TarExtractor.is_extractable(cls, path: Union[Path, str], **kwargs)
TarExtractor.safemembers(members, output_path)
XzExtractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
ZipExtractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])
ZipExtractor.is_extractable(cls, path: Union[Path, str], magic_number: bytes  =  b"")
ZstdExtractor.extract(input_path: Union[Path, str], output_path: Union[Path, str])


repos/datasets/src/datasets/utils/file_utils.py
-------------------------functions----------------------
_add_retries_to_file_obj_read_method(file_obj)
_as_str(path: Union[str, Path, xPath])
_get_extraction_protocol(urlpath: str, download_config: Optional[DownloadConfig]  =  None)
_get_extraction_protocol_with_magic_number(f)
_get_path_extension(path: str)
_prepare_path_and_storage_options(urlpath: str, download_config: Optional[DownloadConfig]  =  None)
_prepare_single_hop_path_and_storage_options(urlpath: str, download_config: Optional[DownloadConfig]  =  None)
_raise_if_offline_mode_is_enabled(msg: Optional[str]  =  None)
_request_with_retry(method: str, url: str, max_retries: int  =  0, base_wait_time: float  =  0.5, max_wait_time: float  =  2, timeout: float  =  10.0, **params, )
add_end_docstrings(*docstr)
add_start_docstrings(*docstr)
cached_path(url_or_filename, download_config = None, **download_kwargs, )
estimate_dataset_size(paths)
fsspec_get(url, temp_file, storage_options = None, desc = None, disable_tqdm = False)
fsspec_head(url, storage_options = None)
ftp_get(url, temp_file, timeout = 10.0)
ftp_head(url, timeout = 10.0)
get_authentication_headers_for_url(url: str, token: Optional[Union[str, bool]]  =  None, use_auth_token: Optional[Union[str, bool]]  =  "deprecated")
get_datasets_user_agent(user_agent: Optional[Union[str, dict]]  =  None)
get_from_cache(url, cache_dir = None, force_download = False, proxies = None, etag_timeout = 100, resume_download = False, user_agent = None, local_files_only = False, use_etag = True, max_retries = 0, token = None, use_auth_token = "deprecated", ignore_url_params = False, storage_options = None, download_desc = None, disable_tqdm = False, )
hash_url_to_filename(url, etag = None)
head_hf_s3(identifier: str, filename: str, use_cdn = False, dataset = True, max_retries = 0)
hf_bucket_url(identifier: str, filename: str, use_cdn = False, dataset = True)
hf_github_url(path: str, name: str, dataset = True, revision: Optional[str]  =  None)
http_get(url, temp_file, proxies = None, resume_size = 0, headers = None, cookies = None, timeout = 100.0, max_retries = 0, desc = None, disable_tqdm = False, )
http_head(url, proxies = None, headers = None, cookies = None, allow_redirects = True, timeout = 10.0, max_retries = 0)
init_hf_modules(hf_modules_cache: Optional[Union[Path, str]]  =  None)
is_local_path(url_or_filename: str)
is_relative_path(url_or_filename: str)
is_remote_url(url_or_filename: str)
readline(f: io.RawIOBase)
relative_to_absolute_path(path: T)
request_etag(url: str, token: Optional[Union[str, bool]]  =  None, use_auth_token: Optional[Union[str, bool]]  =  "deprecated")
stack_multiprocessing_download_progress_bars()
url_or_path_join(base_name: str, *pathnames: str)
url_or_path_parent(url_or_path: str)
xbasename(a)
xdirname(a)
xet_parse(source, parser = None, download_config: Optional[DownloadConfig]  =  None)
xexists(urlpath: str, download_config: Optional[DownloadConfig]  =  None)
xgetsize(path, download_config: Optional[DownloadConfig]  =  None)
xglob(urlpath, *, recursive = False, download_config: Optional[DownloadConfig]  =  None)
xgzip_open(filepath_or_buffer, *args, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xisdir(path, download_config: Optional[DownloadConfig]  =  None)
xisfile(path, download_config: Optional[DownloadConfig]  =  None)
xjoin(a, *p)
xlistdir(path: str, download_config: Optional[DownloadConfig]  =  None)
xnumpy_load(filepath_or_buffer, *args, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xopen(file: str, mode = "r", *args, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xpandas_read_csv(filepath_or_buffer, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xpandas_read_excel(filepath_or_buffer, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xpyarrow_parquet_read_table(filepath_or_buffer, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xrelpath(path, start = None)
xsio_loadmat(filepath_or_buffer, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xsplit(a)
xsplitext(a)
xwalk(urlpath, download_config: Optional[DownloadConfig]  =  None, **kwargs)
xxml_dom_minidom_parse(filename_or_file, download_config: Optional[DownloadConfig]  =  None, **kwargs)

-------------------------methods----------------------
ArchiveIterable._iter_from_fileobj(cls, f)
ArchiveIterable._iter_from_urlpath(cls, urlpath: str, download_config: Optional[DownloadConfig]  =  None)
ArchiveIterable._iter_tar(f)
ArchiveIterable._iter_zip(f)
ArchiveIterable.from_buf(cls, fileobj)
ArchiveIterable.from_urlpath(cls, urlpath_or_buf, download_config: Optional[DownloadConfig]  =  None)
FilesIterable._iter_from_urlpaths(cls, urlpaths: Union[str, List[str]], download_config: Optional[DownloadConfig]  =  None)
FilesIterable.from_urlpaths(cls, urlpaths, download_config: Optional[DownloadConfig]  =  None)
TqdmCallback.__init__(self, tqdm_kwargs = None, *args, **kwargs)
_IterableFromGenerator.__init__(self, generator: Callable, *args, **kwargs)
_IterableFromGenerator.__iter__(self)
xPath.__str__(self)
xPath.__truediv__(self, p: str)
xPath.exists(self, download_config: Optional[DownloadConfig]  =  None)
xPath.glob(self, pattern, download_config: Optional[DownloadConfig]  =  None)
xPath.joinpath(self, *p: Tuple[str, ...])
xPath.name(self)
xPath.open(self, *args, **kwargs)
xPath.parent(self)
xPath.rglob(self, pattern, **kwargs)
xPath.stem(self)
xPath.suffix(self)
xPath.with_suffix(self, suffix)


repos/datasets/src/datasets/utils/filelock.py


repos/datasets/src/datasets/utils/hub.py


repos/datasets/src/datasets/utils/info_utils.py
-------------------------functions----------------------
get_size_checksum_dict(path: str, record_checksum: bool  =  True)
is_small_dataset(dataset_size)
verify_checksums(expected_checksums: Optional[dict], recorded_checksums: dict, verification_name = None)
verify_splits(expected_splits: Optional[dict], recorded_splits: dict)



repos/datasets/src/datasets/utils/logging.py
-------------------------functions----------------------
_configure_library_root_logger()
_get_default_logging_level()
_get_library_name()
_get_library_root_logger()
_reset_library_root_logger()
disable_propagation()
enable_propagation()
get_logger(name: Optional[str]  =  None)
get_verbosity()
set_verbosity(verbosity: int)
set_verbosity_debug()
set_verbosity_error()
set_verbosity_info()
set_verbosity_warning()



repos/datasets/src/datasets/utils/metadata.py
-------------------------functions----------------------
_split_yaml_from_readme(readme_content: str)

-------------------------methods----------------------
DatasetMetadata._to_readme(self, readme_content: Optional[str]  =  None)
DatasetMetadata.from_readme(cls, path: Union[Path, str])
DatasetMetadata.from_yaml_string(cls, string: str)
DatasetMetadata.to_readme(self, path: Path)
DatasetMetadata.to_yaml_string(self)
MetadataConfigs._raise_if_data_files_field_not_valid(metadata_config: dict)
_NoDuplicateSafeLoader._check_no_duplicates_on_constructed_node(self, node)
_NoDuplicateSafeLoader.construct_mapping(self, node, deep = False)


repos/datasets/src/datasets/utils/patching.py
-------------------------methods----------------------
_PatchedModuleObj.__init__(self, module, attrs = None)
patch_submodule.__enter__(self)
patch_submodule.__exit__(self, *exc_info)
patch_submodule.__init__(self, obj, target: str, new, attrs = None)
patch_submodule.start(self)
patch_submodule.stop(self)


repos/datasets/src/datasets/utils/py_utils.py
-------------------------functions----------------------
_convert_github_url(url_path: str)
_get_pool_pid(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool])
_single_map_nested(args)
_write_generator_to_queue(queue: queue.Queue, func: Callable[..., Iterable[Y]], kwargs: dict)
asdict(obj)
convert_file_size_to_int(size: Union[int, str])
copyfunc(func)
first_non_null_value(iterable)
get_imports(file_path: str)
glob_pattern_to_regex(pattern)
has_sufficient_disk_space(needed_bytes, directory = ".")
iflatmap_unordered(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool], func: Callable[..., Iterable[Y]], *, kwargs_iterable: Iterable[dict], )
iter_batched(iterable: Iterable[T], n: int)
lock_importable_file(importable_local_file: str)
map_nested(function: Callable[[Any], Any], data_struct: Any, dict_only: bool  =  False, map_list: bool  =  True, map_tuple: bool  =  False, map_numpy: bool  =  False, num_proc: Optional[int]  =  None, parallel_min_length: int  =  2, batched: bool  =  False, batch_size: Optional[int]  =  1000, types: Optional[tuple]  =  None, disable_tqdm: bool  =  True, desc: Optional[str]  =  None, )
no_op_if_value_is_null(func)
size_str(size_in_bytes)
string_to_dict(string: str, pattern: str)
temp_seed(seed: int, set_pytorch = False, set_tensorflow = False)
temporary_assignment(obj, attr, value)
unique_values(values)
zip_dict(*dicts)

-------------------------methods----------------------
NestedDataStructure.__init__(self, data = None)
NestedDataStructure.flatten(self, data = None)
NonMutableDict.__init__(self, *args, **kwargs)
NonMutableDict.__setitem__(self, key, value)
NonMutableDict.update(self, other)
classproperty.__get__(self, obj, objtype = None)


repos/datasets/src/datasets/utils/readme.py
-------------------------functions----------------------
load_yaml_resource(resource: str)

-------------------------methods----------------------
ReadMe.__init__(self, name: str, lines: List[str], structure: dict  =  None, suppress_parsing_errors: bool  =  False)
ReadMe.__str__(self)
ReadMe._validate(self, readme_structure)
ReadMe.from_readme(cls, path: Path, structure: dict  =  None, suppress_parsing_errors: bool  =  False)
ReadMe.from_string(cls, string: str, structure: dict  =  None, root_name: str  =  "root", suppress_parsing_errors: bool  =  False)
ReadMe.parse(self, suppress_parsing_errors: bool  =  False)
ReadMe.validate(self)
Section.__init__(self, name: str, level: str, lines: List[str]  =  None, suppress_parsing_errors: bool  =  False)
Section.parse(self, suppress_parsing_errors: bool  =  False)
Section.to_dict(self)
Section.validate(self, structure: dict)


repos/datasets/src/datasets/utils/sharding.py
-------------------------functions----------------------
_distribute_shards(num_shards: int, max_num_jobs: int)
_merge_gen_kwargs(gen_kwargs_list: List[dict])
_number_of_shards_in_gen_kwargs(gen_kwargs: dict)
_shuffle_gen_kwargs(rng: np.random.Generator, gen_kwargs: dict)
_split_gen_kwargs(gen_kwargs: dict, max_num_jobs: int)



repos/datasets/src/datasets/utils/stratify.py
-------------------------functions----------------------
approximate_mode(class_counts, n_draws, rng)
stratified_shuffle_split_generate_indices(y, n_train, n_test, rng, n_splits = 10)



repos/datasets/src/datasets/utils/tf_utils.py
-------------------------functions----------------------
dataset_to_tf(dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, output_signature, shuffle, batch_size, drop_remainder, )
is_numeric_feature(feature)
is_numeric_pa_type(pa_type)
minimal_tf_collate_fn(features)
minimal_tf_collate_fn_with_renaming(features)
multiprocess_dataset_to_tf(dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, output_signature, shuffle, batch_size, drop_remainder, num_workers, )
np_get_batch(indices, dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, return_dict = False)

-------------------------methods----------------------
NumpyMultiprocessingGenerator.__call__(self)
NumpyMultiprocessingGenerator.__init__(self, dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, output_signature, shuffle, batch_size, drop_remainder, num_workers, )
NumpyMultiprocessingGenerator.__iter__(self)
NumpyMultiprocessingGenerator.distribute_batches(dataset, batch_size, drop_remainder, num_workers, shuffle)
NumpyMultiprocessingGenerator.worker_loop(dataset, cols_to_retain, collate_fn, collate_fn_args, columns_to_np_types, columns_to_ranks, string_columns, indices, extra_batch, worker_name, array_ready_event, array_loaded_event, )
SharedMemoryContext.__enter__(self)
SharedMemoryContext.__exit__(self, exc_type, exc_value, traceback)
SharedMemoryContext.__init__(self)
SharedMemoryContext.get_array(self, name, shape, dtype, create)
SharedMemoryContext.get_shm(self, name, size, create)


repos/datasets/src/datasets/utils/tqdm.py
-------------------------functions----------------------
are_progress_bars_disabled()
disable_progress_bars()
enable_progress_bars()
is_progress_bar_enabled()

-------------------------methods----------------------
tqdm.__delattr__(self, attr: str)
tqdm.__init__(self, *args, **kwargs)


repos/datasets/src/datasets/utils/track.py
-------------------------methods----------------------
TrackedIterable.__init__(self)
TrackedIterable.__repr__(self)
tracked_list.__init__(self, *args, **kwargs)
tracked_list.__iter__(self)
tracked_list.__repr__(self)
tracked_str.__repr__(self)
tracked_str.get_origin(self)
tracked_str.set_origin(self, origin: str)


repos/datasets/src/datasets/utils/typing.py


repos/datasets/src/datasets/utils/version.py
-------------------------functions----------------------
_str_to_version_tuple(version_str)
_version_tuple_to_str(version_tuple)

-------------------------methods----------------------
Version.__eq__(self, other)
Version.__hash__(self)
Version.__lt__(self, other)
Version.__post_init__(self)
Version.__repr__(self)
Version._to_yaml_string(self)
Version._validate_operand(self, other)
Version.from_dict(cls, dic)
Version.tuple(self)


repos/datasets/templates/new_dataset_script.py
-------------------------methods----------------------
NewDataset._generate_examples(self, filepath, split)
NewDataset._info(self)
NewDataset._split_generators(self, dl_manager)


repos/datasets/tests/__init__.py


repos/datasets/tests/_test_patching.py


repos/datasets/tests/commands/__init__.py


repos/datasets/tests/commands/conftest.py
-------------------------functions----------------------
dataset_loading_script_code()
dataset_loading_script_dir(dataset_loading_script_name, dataset_loading_script_code, tmp_path)
dataset_loading_script_name()

-------------------------methods----------------------
__DummyDataset1__._generate_examples(self, filepath)
__DummyDataset1__._info(self)
__DummyDataset1__._split_generators(self, dl_manager)


repos/datasets/tests/commands/test_test.py
-------------------------functions----------------------
is_1percent_close(source, target)
test_test_command(dataset_loading_script_dir)



repos/datasets/tests/conftest.py
-------------------------functions----------------------
disable_tqdm_output()
pytest_collection_modifyitems(config, items)
pytest_configure(config)
set_sqlalchemy_silence_uber_warning(monkeypatch)
set_test_cache_config(tmp_path_factory, monkeypatch)
set_update_download_counts_to_false(monkeypatch)
zero_time_out_for_remote_code()



repos/datasets/tests/distributed_scripts/run_torch_distributed.py
-------------------------functions----------------------
gen(shards: List[str])
main()



repos/datasets/tests/features/__init__.py


repos/datasets/tests/features/test_array_xd.py
-------------------------functions----------------------
generate_examples(features: dict, num_examples = 100, seq_shapes = None)
get_array_feature_types()
test_array_xd_numpy_arrow_extractor(dtype, dummy_value)
test_array_xd_with_none()
test_array_xd_with_np(seq_type, dtype, shape, feature_class)
test_dataset_map(with_none)
test_table_to_pandas(dtype, dummy_value)

-------------------------methods----------------------
ArrayXDDynamicTest.get_one_col_dataset(self, first_dim_list, fixed_shape)
ArrayXDDynamicTest.get_two_col_datasset(self, first_dim_list, fixed_shape)
ArrayXDDynamicTest.test_iter_dataset(self)
ArrayXDDynamicTest.test_map_dataset(self)
ArrayXDDynamicTest.test_to_numpy(self)
ArrayXDDynamicTest.test_to_pandas(self)
ArrayXDDynamicTest.test_to_pylist(self)
ArrayXDTest._check_getitem_output_type(self, dataset, shape_1, shape_2, first_matrix)
ArrayXDTest.get_dict_example_0(self, shape_1, shape_2)
ArrayXDTest.get_dict_example_1(self, shape_1, shape_2)
ArrayXDTest.get_dict_examples(self, shape_1, shape_2)
ArrayXDTest.get_features(self, array_feature, shape_1, shape_2)
ArrayXDTest.test_from_dict(self, array_feature, shape_1, shape_2)
ArrayXDTest.test_write(self, array_feature, shape_1, shape_2)
ArrayXDTest.test_write_batch(self, array_feature, shape_1, shape_2)
ExtensionTypeCompatibilityTest.test_array2d_nonspecific_shape(self)
ExtensionTypeCompatibilityTest.test_compatability_with_string_values(self)
ExtensionTypeCompatibilityTest.test_extension_indexing(self)
ExtensionTypeCompatibilityTest.test_multiple_extensions_same_row(self)


repos/datasets/tests/features/test_audio.py
-------------------------functions----------------------
iter_archive(archive_path)
jsonl_audio_dataset_path(shared_datadir, tmp_path_factory)
tar_mp3_path(shared_datadir, tmp_path_factory)
tar_wav_path(shared_datadir, tmp_path_factory)
test_audio_decode_example(shared_datadir)
test_audio_decode_example_mp3(shared_datadir)
test_audio_decode_example_opus(shared_datadir)
test_audio_decode_example_pcm(shared_datadir, sampling_rate)
test_audio_embed_storage(shared_datadir)
test_audio_feature_encode_example(shared_datadir, build_example)
test_audio_feature_encode_example_pcm(shared_datadir, build_example)
test_audio_feature_type_to_arrow()
test_audio_instantiation()
test_audio_resampling(shared_datadir)
test_audio_resampling_mp3_different_sampling_rates(shared_datadir)
test_dataset_cast_to_audio_features(shared_datadir, build_data)
test_dataset_concatenate_audio_features(shared_datadir)
test_dataset_concatenate_nested_audio_features(shared_datadir)
test_dataset_with_audio_feature(shared_datadir)
test_dataset_with_audio_feature_loaded_from_cache()
test_dataset_with_audio_feature_map_is_decoded(shared_datadir)
test_dataset_with_audio_feature_map_is_not_decoded(shared_datadir)
test_dataset_with_audio_feature_map_undecoded(shared_datadir)
test_dataset_with_audio_feature_tar_mp3(tar_mp3_path)
test_dataset_with_audio_feature_tar_wav(tar_wav_path)
test_dataset_with_audio_feature_undecoded(shared_datadir)
test_dataset_with_audio_feature_with_none()
test_formatted_dataset_with_audio_feature(shared_datadir)
test_formatted_dataset_with_audio_feature_undecoded(shared_datadir)
test_load_dataset_with_audio_feature(streaming, jsonl_audio_dataset_path, shared_datadir)
test_resampling_after_loading_dataset_with_audio_feature(shared_datadir)
test_resampling_after_loading_dataset_with_audio_feature_mp3(shared_datadir)
test_resampling_at_loading_dataset_with_audio_feature(shared_datadir)
test_resampling_at_loading_dataset_with_audio_feature_mp3(shared_datadir)



repos/datasets/tests/features/test_features.py
-------------------------functions----------------------
dict_diff(d1: dict, d2: dict)
iternumpy(key1, value1, value2)
test_class_label_to_and_from_dict(class_label_arg, tmp_path_factory)
test_classlabel_cast_storage()
test_classlabel_init(tmp_path_factory)
test_classlabel_int2str()
test_classlabel_str2int()
test_dataset_feature_with_none(feature)
test_encode_batch_with_example_with_empty_first_elem()
test_encode_column_dict_with_none()
test_encode_nested_example_sequence_with_none(inner_type)
test_features_alignment(features: Tuple[List[Features], Features])
test_features_to_arrow_schema(features: Features)
test_features_to_dict(features: Features)
test_features_to_yaml_list(features: Features)

-------------------------methods----------------------
CastToPythonObjectsTest.test_cast_to_python_objects_dataframe(self)
CastToPythonObjectsTest.test_cast_to_python_objects_jax(self)
CastToPythonObjectsTest.test_cast_to_python_objects_list(self)
CastToPythonObjectsTest.test_cast_to_python_objects_pandas_timedelta(self)
CastToPythonObjectsTest.test_cast_to_python_objects_pandas_timestamp(self)
CastToPythonObjectsTest.test_cast_to_python_objects_series(self)
CastToPythonObjectsTest.test_cast_to_python_objects_tf(self)
CastToPythonObjectsTest.test_cast_to_python_objects_torch(self)
CastToPythonObjectsTest.test_cast_to_python_objects_tuple(self)
CastToPythonObjectsTest.test_cast_to_python_or_numpy(self)
CastToPythonObjectsTest.test_dont_iterate_over_each_element_in_a_list(self, mocked_cast)
FeaturesTest.test_class_label_feature_with_no_labels(self)
FeaturesTest.test_feature_named_self_as_kwarg(self)
FeaturesTest.test_feature_named_type(self)
FeaturesTest.test_features_dicts_are_synced(self)
FeaturesTest.test_flatten(self)
FeaturesTest.test_flatten_with_sequence(self)
FeaturesTest.test_from_arrow_schema_simple(self)
FeaturesTest.test_from_arrow_schema_with_sequence(self)
FeaturesTest.test_reorder_fields_as(self)
FeaturesTest.test_string_to_arrow_bijection_for_primitive_types(self)


repos/datasets/tests/features/test_image.py
-------------------------functions----------------------
data_dir(shared_datadir, tmp_path)
dataset_loading_script_dir(tmp_path)
iter_archive(archive_path)
tar_jpg_path(shared_datadir, tmp_path_factory)
test_dataset_cast_to_image_features(shared_datadir, build_data)
test_dataset_concatenate_image_features(shared_datadir)
test_dataset_concatenate_nested_image_features(shared_datadir)
test_dataset_with_image_feature(shared_datadir)
test_dataset_with_image_feature_from_np_array()
test_dataset_with_image_feature_from_pil_image(infer_feature, shared_datadir)
test_dataset_with_image_feature_map(shared_datadir)
test_dataset_with_image_feature_map_change_image(shared_datadir)
test_dataset_with_image_feature_map_undecoded(shared_datadir)
test_dataset_with_image_feature_tar_jpg(tar_jpg_path)
test_dataset_with_image_feature_undecoded(shared_datadir)
test_dataset_with_image_feature_with_none()
test_encode_np_array(array, dtype_cast, expected_image_format)
test_formatted_dataset_with_image_feature(shared_datadir)
test_formatted_dataset_with_image_feature_map(shared_datadir)
test_formatted_dataset_with_image_feature_undecoded(shared_datadir)
test_image_change_mode(shared_datadir)
test_image_decode_example(shared_datadir)
test_image_decode_example_with_exif_orientation_tag(shared_datadir)
test_image_embed_storage(shared_datadir)
test_image_feature_encode_example(shared_datadir, build_example)
test_image_feature_type_to_arrow()
test_image_instantiation()
test_load_dataset_with_image_feature(shared_datadir, data_dir, dataset_loading_script_dir, streaming)

-------------------------methods----------------------
__DummyDataset__._generate_examples(self, filepath, **kwargs)
__DummyDataset__._info(self)
__DummyDataset__._split_generators(self, dl_manager)


repos/datasets/tests/fixtures/__init__.py


repos/datasets/tests/fixtures/files.py
-------------------------functions----------------------
arrow_file(tmp_path_factory, dataset)
arrow_path(tmp_path_factory)
audio_file()
bz2_csv_path(csv_path, tmp_path_factory)
bz2_file(tmp_path_factory)
csv2_path(tmp_path_factory)
csv_path(tmp_path_factory)
data_dir_with_hidden_files(tmp_path_factory)
dataset()
dataset_dict()
geoparquet_path(tmp_path_factory)
gz_file(tmp_path_factory)
image_file()
json_dict_of_lists_path(tmp_path_factory)
json_list_of_dicts_path(tmp_path_factory)
jsonl2_path(tmp_path_factory)
jsonl_312_path(tmp_path_factory)
jsonl_gz_path(tmp_path_factory, jsonl_path)
jsonl_path(tmp_path_factory)
jsonl_str_path(tmp_path_factory)
lz4_file(tmp_path_factory)
parquet_path(tmp_path_factory)
seven_zip_file(tmp_path_factory, text_file)
sqlite_path(tmp_path_factory)
tar_file(tmp_path_factory, text_file)
tar_jsonl_path(jsonl_path, jsonl2_path, tmp_path_factory)
tar_nested_jsonl_path(tar_jsonl_path, jsonl_path, jsonl2_path, tmp_path_factory)
text2_path(tmp_path_factory)
text_dir(tmp_path_factory)
text_dir_with_unsupported_extension(tmp_path_factory)
text_file(tmp_path_factory)
text_file_content()
text_gz_path(tmp_path_factory, text_path)
text_path(tmp_path_factory)
text_path_with_unicode_new_lines(tmp_path_factory)
xml_file(tmp_path_factory)
xz_file(tmp_path_factory)
zip_csv_path(csv_path, csv2_path, tmp_path_factory)
zip_csv_with_dir_path(csv_path, csv2_path, tmp_path_factory)
zip_file(tmp_path_factory, text_file)
zip_image_path(image_file, tmp_path_factory)
zip_jsonl_path(jsonl_path, jsonl2_path, tmp_path_factory)
zip_jsonl_with_dir_path(jsonl_path, jsonl2_path, tmp_path_factory)
zip_nested_jsonl_path(zip_jsonl_path, jsonl_path, jsonl2_path, tmp_path_factory)
zip_text_path(text_path, text2_path, tmp_path_factory)
zip_text_with_dir_path(text_path, text2_path, tmp_path_factory)
zip_unsupported_ext_path(text_path, text2_path, tmp_path_factory)
zip_uppercase_csv_path(csv_path, csv2_path, tmp_path_factory)
zstd_file(tmp_path_factory)



repos/datasets/tests/fixtures/fsspec.py
-------------------------functions----------------------
mock_fsspec()
mockfs(tmp_path_factory, mock_fsspec)
tmpfs(tmp_path_factory, mock_fsspec)

-------------------------methods----------------------
MockFileSystem.__init__(self, *args, local_root_dir, **kwargs)
MockFileSystem._open(self, path, *args, **kwargs)
MockFileSystem._strip_protocol(cls, path)
MockFileSystem.cp_file(self, path1, path2, *args, **kwargs)
MockFileSystem.created(self, path)
MockFileSystem.info(self, path, *args, **kwargs)
MockFileSystem.ls(self, path, detail = True, *args, **kwargs)
MockFileSystem.makedirs(self, path, *args, **kwargs)
MockFileSystem.mkdir(self, path, *args, **kwargs)
MockFileSystem.modified(self, path)
MockFileSystem.rm(self, path, *args, **kwargs)
MockFileSystem.rm_file(self, path, *args, **kwargs)
MockFileSystem.rmdir(self, path)
TmpDirFileSystem.__init__(self, *args, **kwargs)
TmpDirFileSystem._strip_protocol(cls, path)


repos/datasets/tests/fixtures/hub.py
-------------------------functions----------------------
ci_hfh_hf_hub_url(monkeypatch)
ci_hub_config(monkeypatch)
cleanup_repo(hf_api)
hf_api()
hf_private_dataset_repo_txt_data(hf_private_dataset_repo_txt_data_, ci_hub_config, ci_hfh_hf_hub_url)
hf_private_dataset_repo_txt_data_(hf_api: HfApi, hf_token, text_file_content)
hf_private_dataset_repo_zipped_img_data(hf_private_dataset_repo_zipped_img_data_, ci_hub_config, ci_hfh_hf_hub_url)
hf_private_dataset_repo_zipped_img_data_(hf_api: HfApi, hf_token, zip_image_path)
hf_private_dataset_repo_zipped_txt_data(hf_private_dataset_repo_zipped_txt_data_, ci_hub_config, ci_hfh_hf_hub_url)
hf_private_dataset_repo_zipped_txt_data_(hf_api: HfApi, hf_token, zip_csv_with_dir_path)
hf_token()
set_ci_hub_access_token(ci_hub_config)
temporary_repo(cleanup_repo)



repos/datasets/tests/io/__init__.py


repos/datasets/tests/io/test_csv.py
-------------------------functions----------------------
_check_csv_dataset(dataset, expected_features)
_check_csv_datasetdict(dataset_dict, expected_features, splits = ("train", )
iter_csv_file(csv_path)
test_csv_datasetdict_reader_features(features, csv_path, tmp_path)
test_csv_datasetdict_reader_keep_in_memory(keep_in_memory, csv_path, tmp_path)
test_csv_datasetdict_reader_split(split, csv_path, tmp_path)
test_dataset_from_csv_features(features, csv_path, tmp_path)
test_dataset_from_csv_keep_in_memory(keep_in_memory, csv_path, tmp_path)
test_dataset_from_csv_path_type(path_type, csv_path, tmp_path)
test_dataset_from_csv_split(split, csv_path, tmp_path)
test_dataset_to_csv(csv_path, tmp_path)
test_dataset_to_csv_fsspec(dataset, mockfs)
test_dataset_to_csv_invalidproc(csv_path, tmp_path)
test_dataset_to_csv_multiproc(csv_path, tmp_path)



repos/datasets/tests/io/test_json.py
-------------------------functions----------------------
_check_json_dataset(dataset, expected_features)
_check_json_datasetdict(dataset_dict, expected_features, splits = ("train", )
load_json(buffer)
load_json_lines(buffer)
test_dataset_from_json_features(features, jsonl_path, tmp_path)
test_dataset_from_json_keep_in_memory(keep_in_memory, jsonl_path, tmp_path)
test_dataset_from_json_path_type(path_type, jsonl_path, tmp_path)
test_dataset_from_json_split(split, jsonl_path, tmp_path)
test_dataset_from_json_with_mismatched_features(jsonl_312_path, tmp_path)
test_dataset_from_json_with_unsorted_column_names(features, jsonl_312_path, tmp_path)
test_datasetdict_from_json_features(features, jsonl_path, tmp_path)
test_datasetdict_from_json_keep_in_memory(keep_in_memory, jsonl_path, tmp_path)
test_datasetdict_from_json_splits(split, jsonl_path, tmp_path)

-------------------------methods----------------------
TestJsonDatasetWriter.test_dataset_to_json_compression(self, shared_datadir, tmp_path_factory, extension, compression, dataset)
TestJsonDatasetWriter.test_dataset_to_json_fsspec(self, dataset, mockfs)
TestJsonDatasetWriter.test_dataset_to_json_lines(self, lines, load_json_function, dataset)
TestJsonDatasetWriter.test_dataset_to_json_lines_multiproc(self, lines, load_json_function, dataset)
TestJsonDatasetWriter.test_dataset_to_json_orient(self, orient, container, keys, len_at, dataset)
TestJsonDatasetWriter.test_dataset_to_json_orient_invalidproc(self, dataset)
TestJsonDatasetWriter.test_dataset_to_json_orient_multiproc(self, orient, container, keys, len_at, dataset)


repos/datasets/tests/io/test_parquet.py
-------------------------functions----------------------
_check_parquet_dataset(dataset, expected_features)
_check_parquet_datasetdict(dataset_dict, expected_features, splits = ("train", )
test_dataset_from_parquet_features(features, parquet_path, tmp_path)
test_dataset_from_parquet_keep_in_memory(keep_in_memory, parquet_path, tmp_path)
test_dataset_from_parquet_path_type(path_type, parquet_path, tmp_path)
test_dataset_from_parquet_split(split, parquet_path, tmp_path)
test_dataset_to_parquet_fsspec(dataset, mockfs)
test_dataset_to_parquet_keeps_features(shared_datadir, tmp_path)
test_get_writer_batch_size(feature, expected)
test_parquet_datasetdict_reader_columns(streaming, columns, pass_features, pass_info, parquet_path, tmp_path)
test_parquet_datasetdict_reader_features(streaming, features, parquet_path, tmp_path)
test_parquet_datasetdict_reader_keep_in_memory(keep_in_memory, parquet_path, tmp_path)
test_parquet_datasetdict_reader_split(split, parquet_path, tmp_path)
test_parquet_read_geoparquet(geoparquet_path, tmp_path)
test_parquet_write(dataset, tmp_path)



repos/datasets/tests/io/test_sql.py
-------------------------functions----------------------
_check_sql_dataset(dataset, expected_features)
iter_sql_file(sqlite_path)
test_dataset_from_sql_features(features, sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning)
test_dataset_from_sql_keep_in_memory(keep_in_memory, sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning)
test_dataset_to_sql(sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning)
test_dataset_to_sql_invalidproc(sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning)
test_dataset_to_sql_multiproc(sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning)



repos/datasets/tests/io/test_text.py
-------------------------functions----------------------
_check_text_dataset(dataset, expected_features)
_check_text_datasetdict(dataset_dict, expected_features, splits = ("train", )
test_dataset_from_text_features(features, text_path, tmp_path)
test_dataset_from_text_keep_in_memory(keep_in_memory, text_path, tmp_path)
test_dataset_from_text_path_type(path_type, text_path, tmp_path)
test_dataset_from_text_split(split, text_path, tmp_path)
test_datasetdict_from_text_features(features, text_path, tmp_path)
test_datasetdict_from_text_keep_in_memory(keep_in_memory, text_path, tmp_path)
test_datasetdict_from_text_split(split, text_path, tmp_path)



repos/datasets/tests/packaged_modules/__init__.py


repos/datasets/tests/packaged_modules/test_audiofolder.py
-------------------------functions----------------------
audio_file_with_metadata(tmp_path, audio_file)
audio_files_with_labels_and_duplicated_label_key_in_metadata(tmp_path, audio_file)
audio_files_with_metadata_that_misses_one_audio(tmp_path, audio_file)
cache_dir(tmp_path)
data_files_with_labels_no_metadata(tmp_path, audio_file)
data_files_with_one_split_and_metadata(tmp_path, audio_file)
data_files_with_two_splits_and_metadata(request, tmp_path, audio_file)
data_files_with_zip_archives(tmp_path, audio_file)
test_data_files_with_metadata_and_archives(streaming, cache_dir, data_files_with_zip_archives)
test_data_files_with_metadata_and_multiple_splits(streaming, cache_dir, data_files_with_two_splits_and_metadata)
test_data_files_with_metadata_and_single_split(streaming, cache_dir, data_files_with_one_split_and_metadata)
test_data_files_with_with_metadata_in_different_formats(cache_dir, tmp_path, audio_file)
test_data_files_with_wrong_audio_file_name_column_in_metadata_file(cache_dir, tmp_path, audio_file)
test_data_files_with_wrong_metadata_file_name(cache_dir, tmp_path, audio_file)
test_generate_examples_drop_labels(data_files_with_labels_no_metadata, drop_metadata, drop_labels)
test_generate_examples_drop_metadata(audio_file_with_metadata, drop_metadata, drop_labels)
test_generate_examples_duplicated_label_key(audio_files_with_labels_and_duplicated_label_key_in_metadata, drop_metadata, drop_labels, cache_dir, caplog)
test_generate_examples_with_labels(data_files_with_labels_no_metadata, cache_dir)
test_generate_examples_with_metadata_in_wrong_location(audio_file, audio_file_with_metadata, drop_metadata)
test_generate_examples_with_metadata_that_misses_one_audio(audio_files_with_metadata_that_misses_one_audio, drop_metadata)



repos/datasets/tests/packaged_modules/test_cache.py
-------------------------functions----------------------
test_cache(text_dir: Path, tmp_path: Path)
test_cache_auto_hash(text_dir: Path, tmp_path: Path)
test_cache_auto_hash_with_custom_config(text_dir: Path, tmp_path: Path)
test_cache_capital_letters(tmp_path: Path)
test_cache_missing(text_dir: Path, tmp_path: Path)
test_cache_multi_configs(tmp_path: Path)
test_cache_single_config(tmp_path: Path)
test_cache_streaming(text_dir: Path, tmp_path: Path)



repos/datasets/tests/packaged_modules/test_csv.py
-------------------------functions----------------------
csv_file(tmp_path)
csv_file_with_image(tmp_path, image_file)
csv_file_with_int_list(tmp_path)
malformed_csv_file(tmp_path)
test_csv_cast_image(csv_file_with_image)
test_csv_cast_label(csv_file_with_label)
test_csv_convert_int_list(csv_file_with_int_list)
test_csv_generate_tables_raises_error_with_malformed_csv(csv_file, malformed_csv_file, caplog)



repos/datasets/tests/packaged_modules/test_folder_based_builder.py
-------------------------functions----------------------
auto_text_file(text_file)
cache_dir(tmp_path)
data_files_with_different_levels_no_metadata(tmp_path, auto_text_file)
data_files_with_labels_no_metadata(tmp_path, auto_text_file)
data_files_with_one_label_no_metadata(tmp_path, auto_text_file)
data_files_with_one_split_and_metadata(tmp_path, auto_text_file)
data_files_with_two_splits_and_metadata(tmp_path, auto_text_file)
data_files_with_zip_archives(tmp_path, auto_text_file)
file_with_metadata(tmp_path, text_file)
files_with_labels_and_duplicated_label_key_in_metadata(tmp_path, auto_text_file)
files_with_metadata_that_misses_one_sample(tmp_path, auto_text_file)
test_data_files_with_different_levels_no_metadata(data_files_with_different_levels_no_metadata, drop_labels, remote, cache_dir)
test_data_files_with_metadata_and_archives(streaming, cache_dir, data_files_with_zip_archives)
test_data_files_with_metadata_and_splits(streaming, cache_dir, n_splits, data_files_with_one_split_and_metadata, data_files_with_two_splits_and_metadata)
test_data_files_with_metadata_that_misses_one_sample(files_with_metadata_that_misses_one_sample, drop_metadata, cache_dir)
test_data_files_with_one_label_no_metadata(data_files_with_one_label_no_metadata, drop_labels, remote, cache_dir)
test_data_files_with_wrong_file_name_column_in_metadata_file(cache_dir, tmp_path, auto_text_file)
test_data_files_with_wrong_metadata_file_name(cache_dir, tmp_path, auto_text_file)
test_default_folder_builder_not_usable(data_files_with_labels_no_metadata, cache_dir)
test_generate_examples_drop_labels(data_files_with_labels_no_metadata, auto_text_file, drop_metadata, drop_labels, cache_dir)
test_generate_examples_drop_metadata(file_with_metadata, drop_metadata, drop_labels, cache_dir)
test_generate_examples_duplicated_label_key(files_with_labels_and_duplicated_label_key_in_metadata, drop_metadata, drop_labels, cache_dir, caplog)
test_inferring_labels_from_data_dirs(data_files_with_labels_no_metadata, cache_dir)
test_streaming_patched()



repos/datasets/tests/packaged_modules/test_imagefolder.py
-------------------------functions----------------------
cache_dir(tmp_path)
data_files_with_labels_no_metadata(tmp_path, image_file)
data_files_with_one_split_and_metadata(request, tmp_path, image_file)
data_files_with_two_splits_and_metadata(request, tmp_path, image_file)
data_files_with_zip_archives(tmp_path, image_file)
image_file_with_metadata(tmp_path, image_file)
image_files_with_labels_and_duplicated_label_key_in_metadata(tmp_path, image_file)
image_files_with_metadata_that_misses_one_image(tmp_path, image_file)
test_data_files_with_metadata_and_archives(streaming, cache_dir, data_files_with_zip_archives)
test_data_files_with_metadata_and_multiple_splits(streaming, cache_dir, data_files_with_two_splits_and_metadata)
test_data_files_with_metadata_and_single_split(streaming, cache_dir, data_files_with_one_split_and_metadata)
test_data_files_with_with_metadata_in_different_formats(cache_dir, tmp_path, image_file)
test_data_files_with_wrong_image_file_name_column_in_metadata_file(cache_dir, tmp_path, image_file)
test_data_files_with_wrong_metadata_file_name(cache_dir, tmp_path, image_file)
test_generate_examples_drop_labels(data_files_with_labels_no_metadata, drop_metadata, drop_labels)
test_generate_examples_drop_metadata(image_file_with_metadata, drop_metadata, drop_labels)
test_generate_examples_duplicated_label_key(image_files_with_labels_and_duplicated_label_key_in_metadata, drop_metadata, drop_labels, cache_dir, caplog)
test_generate_examples_with_labels(data_files_with_labels_no_metadata, cache_dir)
test_generate_examples_with_metadata_in_wrong_location(image_file, image_file_with_metadata, drop_metadata)
test_generate_examples_with_metadata_that_misses_one_image(image_files_with_metadata_that_misses_one_image, drop_metadata)



repos/datasets/tests/packaged_modules/test_json.py
-------------------------functions----------------------
json_file_with_list_of_dicts(tmp_path)
json_file_with_list_of_dicts_field(tmp_path)
json_file_with_list_of_strings(tmp_path)
jsonl_file(tmp_path)
jsonl_file_utf16_encoded(tmp_path)
test_json_generate_tables(file_fixture, config_kwargs, request)
test_json_generate_tables_with_missing_features(file_fixture, config_kwargs, request)



repos/datasets/tests/packaged_modules/test_spark.py
-------------------------functions----------------------
_get_expected_row_ids_and_row_dicts_for_partition_order(df, partition_order)
test_generate_iterable_examples()
test_repartition_df_if_needed()
test_repartition_df_if_needed_max_num_df_rows()
test_spark_examples_iterable()
test_spark_examples_iterable_shard()
test_spark_examples_iterable_shuffle()



repos/datasets/tests/packaged_modules/test_text.py
-------------------------functions----------------------
test_text_cast_image(text_file_with_image)
test_text_linebreaks(text_file, keep_linebreaks)
test_text_sample_by(sample_by, text_file)
text_file(tmp_path)
text_file_with_image(tmp_path, image_file)



repos/datasets/tests/packaged_modules/test_webdataset.py
-------------------------functions----------------------
audio_wds_file(tmp_path, audio_file)
bad_wds_file(tmp_path, image_file, text_file)
image_wds_file(tmp_path, image_file)
test_audio_webdataset(audio_wds_file)
test_image_webdataset(image_wds_file)
test_webdataset_errors_on_bad_file(bad_wds_file)
test_webdataset_with_features(image_wds_file)



repos/datasets/tests/test_arrow_dataset.py
-------------------------functions----------------------
_check_csv_dataset(dataset, expected_features)
_check_generator_dataset(dataset, expected_features)
_check_json_dataset(dataset, expected_features)
_check_parquet_dataset(dataset, expected_features)
_check_sql_dataset(dataset, expected_features)
_check_text_dataset(dataset, expected_features)
assert_arrow_metadata_are_synced_with_dataset_features(dataset: Dataset)
data_generator()
picklable_filter_function(x)
picklable_filter_function_with_rank(x, r)
picklable_map_function(x)
picklable_map_function_with_indices(x, i)
picklable_map_function_with_indices_and_rank(x, i, r)
picklable_map_function_with_rank(x, r)
test_build_local_temp_path(uri_or_path)
test_cast_with_sliced_list()
test_class_encode_column_with_none(include_nulls)
test_concatenate_datasets(dataset_type, axis, expected_shape, dataset_dict, arrow_path)
test_concatenate_datasets_complex_features(axis)
test_concatenate_datasets_duplicate_columns(dataset)
test_concatenate_datasets_new_columns()
test_concatenate_datasets_with_concatenation_tables(axis, expected_shape, other_dataset_type, dataset_dict, arrow_path)
test_dataset_add_column(column, expected_dtype, in_memory, transform, dataset_dict, arrow_path)
test_dataset_add_item(item, in_memory, dataset_dict, arrow_path, transform)
test_dataset_add_item_introduce_feature_type()
test_dataset_add_item_new_columns()
test_dataset_estimate_nbytes()
test_dataset_filter_batched_indices()
test_dataset_format_with_unformatted_image()
test_dataset_from_csv_features(features, csv_path, tmp_path)
test_dataset_from_csv_keep_in_memory(keep_in_memory, csv_path, tmp_path)
test_dataset_from_csv_path_type(path_type, csv_path, tmp_path)
test_dataset_from_csv_split(split, csv_path, tmp_path)
test_dataset_from_file(in_memory, dataset, arrow_file)
test_dataset_from_generator_features(features, data_generator, tmp_path)
test_dataset_from_generator_keep_in_memory(keep_in_memory, data_generator, tmp_path)
test_dataset_from_json_features(features, jsonl_path, tmp_path)
test_dataset_from_json_keep_in_memory(keep_in_memory, jsonl_path, tmp_path)
test_dataset_from_json_path_type(path_type, jsonl_path, tmp_path)
test_dataset_from_json_split(split, jsonl_path, tmp_path)
test_dataset_from_json_with_class_label_feature(jsonl_str_path, tmp_path)
test_dataset_from_parquet_features(features, parquet_path, tmp_path)
test_dataset_from_parquet_keep_in_memory(keep_in_memory, parquet_path, tmp_path)
test_dataset_from_parquet_path_type(path_type, parquet_path, tmp_path)
test_dataset_from_parquet_split(split, parquet_path, tmp_path)
test_dataset_from_sql_con_type(con_type, sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning, caplog)
test_dataset_from_sql_features(features, sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning)
test_dataset_from_sql_keep_in_memory(keep_in_memory, sqlite_path, tmp_path, set_sqlalchemy_silence_uber_warning)
test_dataset_from_text_features(features, text_path, tmp_path)
test_dataset_from_text_keep_in_memory(keep_in_memory, text_path, tmp_path)
test_dataset_from_text_path_type(path_type, text_path, tmp_path)
test_dataset_from_text_split(split, text_path, tmp_path)
test_dataset_getitem_raises()
test_dataset_iter_batch(batch_size, drop_last_batch)
test_dataset_to_iterable_dataset(dataset: Dataset)
test_dataset_to_json(dataset, tmp_path)
test_dataset_with_torch_dataloader(dataset, batch_size)
test_dummy_dataset_serialize_fs(dataset, mockfs)
test_from_spark()
test_from_spark_different_cache()
test_from_spark_features()
test_interleave_datasets()
test_interleave_datasets_oversampling_strategy()
test_interleave_datasets_probabilities()
test_interleave_datasets_probabilities_oversampling_strategy()
test_map_cases(return_lazy_dict)
test_pickle_dataset_after_transforming_the_table(in_memory, method_and_params, arrow_file)
test_sort_with_none(null_placement)
test_update_metadata_with_features(dataset_dict)

-------------------------methods----------------------
BaseDatasetTest._create_dummy_dataset(self, in_memory: bool, tmp_dir: str, multiple_columns = False, array_features = False, nested_features = False)
BaseDatasetTest._to(self, in_memory, tmp_dir, *datasets)
BaseDatasetTest.inject_fixtures(self, caplog, set_sqlalchemy_silence_uber_warning)
BaseDatasetTest.test_cast(self, in_memory)
BaseDatasetTest.test_class_encode_column(self, in_memory)
BaseDatasetTest.test_concatenate(self, in_memory)
BaseDatasetTest.test_concatenate_formatted(self, in_memory)
BaseDatasetTest.test_concatenate_pickle(self, in_memory)
BaseDatasetTest.test_concatenate_with_indices(self, in_memory)
BaseDatasetTest.test_concatenate_with_indices_from_disk(self, in_memory)
BaseDatasetTest.test_dataset_getitem(self, in_memory)
BaseDatasetTest.test_dummy_dataset(self, in_memory)
BaseDatasetTest.test_dummy_dataset_deepcopy(self, in_memory)
BaseDatasetTest.test_dummy_dataset_load_from_disk(self, in_memory)
BaseDatasetTest.test_dummy_dataset_pickle(self, in_memory)
BaseDatasetTest.test_dummy_dataset_serialize(self, in_memory)
BaseDatasetTest.test_export(self, in_memory)
BaseDatasetTest.test_filter(self, in_memory)
BaseDatasetTest.test_filter_batched(self, in_memory)
BaseDatasetTest.test_filter_caching(self, in_memory)
BaseDatasetTest.test_filter_empty(self, in_memory)
BaseDatasetTest.test_filter_fn_kwargs(self, in_memory)
BaseDatasetTest.test_filter_input_columns(self, in_memory)
BaseDatasetTest.test_filter_multiprocessing(self, in_memory)
BaseDatasetTest.test_filter_with_indices_mapping(self, in_memory)
BaseDatasetTest.test_flatten(self, in_memory)
BaseDatasetTest.test_flatten_complex_image(self, in_memory)
BaseDatasetTest.test_flatten_indices(self, in_memory)
BaseDatasetTest.test_format_nested(self, in_memory)
BaseDatasetTest.test_format_pandas(self, in_memory)
BaseDatasetTest.test_format_polars(self, in_memory)
BaseDatasetTest.test_format_ragged_vectors(self, in_memory)
BaseDatasetTest.test_format_vectors(self, in_memory)
BaseDatasetTest.test_keep_features_after_loading_from_cache(self, in_memory)
BaseDatasetTest.test_keep_features_after_transform_specified(self, in_memory)
BaseDatasetTest.test_keep_features_after_transform_to_file(self, in_memory)
BaseDatasetTest.test_keep_features_after_transform_to_memory(self, in_memory)
BaseDatasetTest.test_keep_features_after_transform_unspecified(self, in_memory)
BaseDatasetTest.test_keep_features_with_new_features(self, in_memory)
BaseDatasetTest.test_map(self, in_memory)
BaseDatasetTest.test_map_batched(self, in_memory)
BaseDatasetTest.test_map_caching(self, in_memory)
BaseDatasetTest.test_map_crash_subprocess(self, in_memory)
BaseDatasetTest.test_map_fn_kwargs(self, in_memory)
BaseDatasetTest.test_map_input_columns(self, in_memory)
BaseDatasetTest.test_map_jax(self, in_memory)
BaseDatasetTest.test_map_multiprocessing(self, in_memory)
BaseDatasetTest.test_map_nested(self, in_memory)
BaseDatasetTest.test_map_new_features(self, in_memory)
BaseDatasetTest.test_map_numpy(self, in_memory)
BaseDatasetTest.test_map_remove_columns(self, in_memory)
BaseDatasetTest.test_map_return_example_as_dict_value(self, in_memory)
BaseDatasetTest.test_map_return_pa_table(self, in_memory)
BaseDatasetTest.test_map_return_pd_dataframe(self, in_memory)
BaseDatasetTest.test_map_stateful_callable(self, in_memory)
BaseDatasetTest.test_map_tensor_batched(self, in_memory)
BaseDatasetTest.test_map_tf(self, in_memory)
BaseDatasetTest.test_map_torch(self, in_memory)
BaseDatasetTest.test_pickle_after_many_transforms_on_disk(self, in_memory)
BaseDatasetTest.test_remove_columns(self, in_memory)
BaseDatasetTest.test_rename_column(self, in_memory)
BaseDatasetTest.test_rename_columns(self, in_memory)
BaseDatasetTest.test_restore_saved_format(self, in_memory)
BaseDatasetTest.test_select(self, in_memory)
BaseDatasetTest.test_select_columns(self, in_memory)
BaseDatasetTest.test_select_then_map(self, in_memory)
BaseDatasetTest.test_set_format_numpy_multiple_columns(self, in_memory)
BaseDatasetTest.test_set_format_pandas(self, in_memory)
BaseDatasetTest.test_set_format_polars(self, in_memory)
BaseDatasetTest.test_set_format_tf(self, in_memory)
BaseDatasetTest.test_set_format_torch(self, in_memory)
BaseDatasetTest.test_set_transform(self, in_memory)
BaseDatasetTest.test_shard(self, in_memory)
BaseDatasetTest.test_shuffle(self, in_memory)
BaseDatasetTest.test_sort(self, in_memory)
BaseDatasetTest.test_tf_dataset_conversion(self, in_memory)
BaseDatasetTest.test_tf_dataset_options(self, in_memory)
BaseDatasetTest.test_tf_index_reshuffling(self, in_memory)
BaseDatasetTest.test_tf_label_renaming(self, in_memory)
BaseDatasetTest.test_to_csv(self, in_memory)
BaseDatasetTest.test_to_dict(self, in_memory)
BaseDatasetTest.test_to_list(self, in_memory)
BaseDatasetTest.test_to_pandas(self, in_memory)
BaseDatasetTest.test_to_parquet(self, in_memory)
BaseDatasetTest.test_to_polars(self, in_memory)
BaseDatasetTest.test_to_sql(self, in_memory)
BaseDatasetTest.test_train_test_split(self, in_memory)
BaseDatasetTest.test_transmit_format(self, in_memory)
BaseDatasetTest.test_transmit_format_dict(self, in_memory)
BaseDatasetTest.test_transmit_format_single(self, in_memory)
BaseDatasetTest.test_with_format(self, in_memory)
BaseDatasetTest.test_with_transform(self, in_memory)
MiscellaneousDatasetTest.test_concatenate_mixed_memory_and_disk(self)
MiscellaneousDatasetTest.test_from_dict(self)
MiscellaneousDatasetTest.test_from_pandas(self)
MiscellaneousDatasetTest.test_from_polars(self)
MiscellaneousDatasetTest.test_set_format_encode(self)
MiscellaneousDatasetTest.test_tf_string_encoding(self)
PickableMagicMock.__reduce__(self)
StratifiedTest.test_errors_train_test_split_stratify(self)
StratifiedTest.test_train_test_split_startify(self)
TaskTemplatesTest.test_align_labels_with_mapping_classification(self)
TaskTemplatesTest.test_align_labels_with_mapping_ner(self)
TaskTemplatesTest.test_concatenate_with_equal_task_templates(self)
TaskTemplatesTest.test_concatenate_with_mixed_task_templates_in_common(self)
TaskTemplatesTest.test_concatenate_with_no_mixed_task_templates_in_common(self)
TaskTemplatesTest.test_concatenate_with_no_task_templates(self)
TaskTemplatesTest.test_task_automatic_speech_recognition(self)
TaskTemplatesTest.test_task_question_answering(self)
TaskTemplatesTest.test_task_summarization(self)
TaskTemplatesTest.test_task_templates_empty_after_preparation(self)
TaskTemplatesTest.test_task_text_classification(self)
TaskTemplatesTest.test_task_text_classification_when_columns_removed(self)
TaskTemplatesTest.test_task_with_incompatible_templates(self)
TaskTemplatesTest.test_task_with_multiple_compatible_task_templates(self)
TaskTemplatesTest.test_task_with_no_template(self)
Unpicklable.__getstate__(self)
Unpicklable.__init__(self, **kwargs)


repos/datasets/tests/test_arrow_reader.py
-------------------------functions----------------------
test_make_file_instructions(split_name, instruction, shard_lengths, read_range)
test_make_file_instructions_basic()
test_make_file_instructions_raises(name, expected_exception)
test_read_files(in_memory, dataset, arrow_file)
test_read_instruction_spec()
test_read_table(in_memory, dataset, arrow_file)

-------------------------methods----------------------
BaseReaderTest.test_read(self)
BaseReaderTest.test_read_files(self)
BaseReaderTest.test_read_sharded(self)
ReaderTest._get_table_from_filename(self, filename_skip_take, in_memory = False)


repos/datasets/tests/test_arrow_writer.py
-------------------------functions----------------------
_check_output(output, expected_num_chunks: int)
change_first_primitive_element_in_list(lst, value)
get_base_dtype(arr_type)
test_always_nullable()
test_arrow_writer_closes_stream(raise_exception, tmp_path)
test_arrow_writer_with_filesystem(mockfs)
test_duplicate_keys(writer_batch_size)
test_key_datatype(writer_batch_size)
test_optimized_int_type_for_typed_sequence(sequence, optimized_int_type, expected_dtype)
test_optimized_typed_sequence(sequence, col, expected_dtype)
test_parquet_writer_write()
test_write(fields, writer_batch_size)
test_write_batch(fields, writer_batch_size)
test_write_file()
test_write_row(fields, writer_batch_size)
test_write_table(fields, writer_batch_size)
test_write_with_features()
test_write_with_keys(writer_batch_size)
test_writer_embed_local_files(tmp_path, embed_local_files)

-------------------------methods----------------------
TypedSequenceTest.test_array_type_forbidden(self)
TypedSequenceTest.test_compatible_extension_type(self)
TypedSequenceTest.test_compatible_type(self)
TypedSequenceTest.test_exhaustive_cast(self)
TypedSequenceTest.test_incompatible_extension_type(self)
TypedSequenceTest.test_incompatible_type(self)
TypedSequenceTest.test_no_type(self)
TypedSequenceTest.test_try_compatible_extension_type(self)
TypedSequenceTest.test_try_compatible_type(self)
TypedSequenceTest.test_try_incompatible_extension_type(self)
TypedSequenceTest.test_try_incompatible_type(self)
TypedSequenceTest.test_try_type_and_type_forbidden(self)


repos/datasets/tests/test_beam.py
-------------------------functions----------------------
get_test_dummy_examples()
get_test_nested_examples()

-------------------------methods----------------------
BeamBuilderTest.test_download_and_prepare(self)
BeamBuilderTest.test_download_and_prepare_sharded(self)
BeamBuilderTest.test_nested_features(self)
BeamBuilderTest.test_no_beam_options(self)
DummyBeamDataset._build_pcollection(self, pipeline, examples)
DummyBeamDataset._info(self)
DummyBeamDataset._split_generators(self, dl_manager, pipeline)
NestedBeamDataset._build_pcollection(self, pipeline, examples)
NestedBeamDataset._info(self)
NestedBeamDataset._split_generators(self, dl_manager, pipeline)


repos/datasets/tests/test_builder.py
-------------------------functions----------------------
_run_concurrent_download_and_prepare(tmp_dir)
_run_test_builder_streaming_works_in_subprocesses(builder)
check_streaming(builder)
test_arrow_based_builder_download_and_prepare_as_parquet(tmp_path)
test_arrow_based_builder_download_and_prepare_sharded(tmp_path)
test_arrow_based_builder_download_and_prepare_with_ambiguous_shards(num_proc, expectation, tmp_path)
test_arrow_based_builder_download_and_prepare_with_max_shard_size(tmp_path)
test_arrow_based_builder_download_and_prepare_with_num_proc(tmp_path)
test_arrow_based_download_and_prepare(tmp_path)
test_beam_based_as_dataset(tmp_path)
test_beam_based_builder_as_streaming_dataset(tmp_path)
test_beam_based_builder_download_and_prepare_as_parquet(tmp_path)
test_beam_based_download_and_prepare(tmp_path)
test_builder_as_dataset(split, expected_dataset_class, expected_dataset_length, in_memory, tmp_path)
test_builder_as_streaming_dataset(tmp_path)
test_builder_config_version(builder_class, kwargs, tmp_path)
test_builder_download_and_prepare_with_absolute_output_dir(tmp_path)
test_builder_download_and_prepare_with_relative_output_dir()
test_builder_streaming_works_in_subprocess(tmp_path)
test_builder_with_filesystem_download_and_prepare(tmp_path, mockfs)
test_builder_with_filesystem_download_and_prepare_reload(tmp_path, mockfs, caplog)
test_custom_writer_batch_size(tmp_path, writer_batch_size, default_writer_batch_size, expected_chunks)
test_generator_based_builder_as_dataset(in_memory, tmp_path)
test_generator_based_builder_download_and_prepare_as_parquet(tmp_path)
test_generator_based_builder_download_and_prepare_sharded(tmp_path)
test_generator_based_builder_download_and_prepare_with_ambiguous_shards(num_proc, expectation, tmp_path)
test_generator_based_builder_download_and_prepare_with_max_shard_size(tmp_path)
test_generator_based_builder_download_and_prepare_with_num_proc(tmp_path)

-------------------------methods----------------------
BuilderTest.test_as_dataset_with_post_process(self)
BuilderTest.test_as_dataset_with_post_process_with_index(self)
BuilderTest.test_cache_dir_for_config_kwargs(self)
BuilderTest.test_cache_dir_for_configured_builder(self)
BuilderTest.test_cache_dir_for_data_dir(self)
BuilderTest.test_cache_dir_for_data_files(self)
BuilderTest.test_cache_dir_for_features(self)
BuilderTest.test_cache_dir_no_args(self)
BuilderTest.test_concurrent_download_and_prepare(self)
BuilderTest.test_config_names(self)
BuilderTest.test_download_and_prepare(self)
BuilderTest.test_download_and_prepare_checksum_computation(self)
BuilderTest.test_download_and_prepare_with_base_path(self)
BuilderTest.test_download_and_prepare_with_post_process(self)
BuilderTest.test_error_download_and_prepare(self)
BuilderTest.test_generator_based_download_and_prepare(self)
CustomBuilderConfig.__init__(self, date = None, language = None, version = "2.0.0", **kwargs)
DummyArrowBasedBuilder._generate_tables(self)
DummyArrowBasedBuilder._info(self)
DummyArrowBasedBuilder._split_generators(self, dl_manager)
DummyArrowBasedBuilderWithAmbiguousShards._generate_tables(self, filepaths, dummy_kwarg_with_different_length)
DummyArrowBasedBuilderWithAmbiguousShards._info(self)
DummyArrowBasedBuilderWithAmbiguousShards._split_generators(self, dl_manager)
DummyArrowBasedBuilderWithShards._generate_tables(self, filepaths)
DummyArrowBasedBuilderWithShards._info(self)
DummyArrowBasedBuilderWithShards._split_generators(self, dl_manager)
DummyBeamBasedBuilder._build_pcollection(self, pipeline)
DummyBeamBasedBuilder._info(self)
DummyBeamBasedBuilder._split_generators(self, dl_manager)
DummyBuilder._info(self)
DummyBuilder._prepare_split(self, split_generator, **kwargs)
DummyBuilder._split_generators(self, dl_manager)
DummyBuilderWithBuilderConfigs._generate_examples(self)
DummyBuilderWithBuilderConfigs._info(self)
DummyBuilderWithBuilderConfigs._split_generators(self, dl_manager)
DummyBuilderWithCustomBuilderConfigs._generate_examples(self)
DummyBuilderWithCustomBuilderConfigs._info(self)
DummyBuilderWithCustomBuilderConfigs._split_generators(self, dl_manager)
DummyBuilderWithDownload.__init__(self, *args, rel_path = None, abs_path = None, **kwargs)
DummyBuilderWithDownload._split_generators(self, dl_manager)
DummyBuilderWithManualDownload._split_generators(self, dl_manager)
DummyBuilderWithManualDownload.manual_download_instructions(self)
DummyBuilderWithVersion._generate_examples(self)
DummyBuilderWithVersion._info(self)
DummyBuilderWithVersion._split_generators(self, dl_manager)
DummyGeneratorBasedBuilder._generate_examples(self)
DummyGeneratorBasedBuilder._info(self)
DummyGeneratorBasedBuilder._split_generators(self, dl_manager)
DummyGeneratorBasedBuilderConfig.__init__(self, content = "foo", times = 2, *args, **kwargs)
DummyGeneratorBasedBuilderWithAmbiguousShards._generate_examples(self, filepaths, dummy_kwarg_with_different_length)
DummyGeneratorBasedBuilderWithAmbiguousShards._info(self)
DummyGeneratorBasedBuilderWithAmbiguousShards._split_generators(self, dl_manager)
DummyGeneratorBasedBuilderWithConfig._generate_examples(self)
DummyGeneratorBasedBuilderWithConfig._info(self)
DummyGeneratorBasedBuilderWithConfig._split_generators(self, dl_manager)
DummyGeneratorBasedBuilderWithIntegers._generate_examples(self)
DummyGeneratorBasedBuilderWithIntegers._info(self)
DummyGeneratorBasedBuilderWithIntegers._split_generators(self, dl_manager)
DummyGeneratorBasedBuilderWithShards._generate_examples(self, filepaths)
DummyGeneratorBasedBuilderWithShards._info(self)
DummyGeneratorBasedBuilderWithShards._split_generators(self, dl_manager)


repos/datasets/tests/test_data_files.py
-------------------------functions----------------------
complex_data_dir(tmp_path)
dummy_fs()
hub_dataset_repo_path(tmpfs, complex_data_dir)
hub_dataset_repo_patterns_results(hub_dataset_repo_path, complex_data_dir, pattern_results)
is_relative_to(path, *other)
mock_fs(file_paths: List[str])
pattern_results(complex_data_dir)
test_DataFilesDict_from_patterns_in_dataset_repository(hub_dataset_repo_path, hub_dataset_repo_patterns_results, pattern)
test_DataFilesDict_from_patterns_in_dataset_repository_hashing(hub_dataset_repo_path)
test_DataFilesDict_from_patterns_in_dataset_repository_with_base_path(hub_dataset_repo_path, pattern, size, base_path, split_name)
test_DataFilesDict_from_patterns_locally(complex_data_dir, pattern_results, pattern)
test_DataFilesDict_from_patterns_locally_or_remote_hashing(text_file)
test_DataFilesList_from_patterns_in_dataset_repository_(hub_dataset_repo_path, hub_dataset_repo_patterns_results, pattern)
test_DataFilesList_from_patterns_locally_with_extra_files(complex_data_dir, text_file)
test_DataFilesList_from_patterns_raises_FileNotFoundError(complex_data_dir)
test_DataFilesPatternsDict(text_file)
test_DataFilesPatternsList(text_file)
test_fail_resolve_pattern_in_dataset_repository(hub_dataset_repo_path)
test_fail_resolve_pattern_locally(complex_data_dir)
test_get_data_files_patterns(base_path, data_file_per_split)
test_get_data_patterns_from_directory_with_the_word_data_twice(tmp_path)
test_get_metadata_files_patterns(metadata_files)
test_is_inside_unrequested_special_dir(complex_data_dir, pattern_results)
test_is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(complex_data_dir, pattern_results)
test_pattern_results_fixture(pattern_results, pattern)
test_resolve_pattern_fs(dummy_fs)
test_resolve_pattern_in_dataset_repository(hub_dataset_repo_path, pattern, hub_dataset_repo_patterns_results)
test_resolve_pattern_in_dataset_repository_hidden_base_path(tmpfs)
test_resolve_pattern_in_dataset_repository_returns_hidden_dir_only_if_requested(hub_dataset_repo_path)
test_resolve_pattern_in_dataset_repository_returns_hidden_file_only_if_requested(hub_dataset_repo_path)
test_resolve_pattern_in_dataset_repository_returns_special_dir_only_if_requested(hub_dataset_repo_path)
test_resolve_pattern_in_dataset_repository_special_base_path(tmpfs)
test_resolve_pattern_in_dataset_repository_with_base_path(hub_dataset_repo_path, pattern, size, base_path)
test_resolve_pattern_in_dataset_repository_with_extensions(hub_dataset_repo_path, pattern, size, extensions)
test_resolve_pattern_locally(complex_data_dir, pattern, pattern_results)
test_resolve_pattern_locally_does_not_resolve_symbolic_links(tmp_path, complex_data_dir)
test_resolve_pattern_locally_hidden_base_path(tmp_path)
test_resolve_pattern_locally_returns_hidden_file_only_if_requested(complex_data_dir)
test_resolve_pattern_locally_returns_special_dir_only_if_requested(complex_data_dir)
test_resolve_pattern_locally_sorted_files(tmp_path_factory)
test_resolve_pattern_locally_special_base_path(tmp_path)
test_resolve_pattern_locally_with_absolute_path(tmp_path, complex_data_dir)
test_resolve_pattern_locally_with_dot_in_base_path(complex_data_dir)
test_resolve_pattern_locally_with_double_dots(tmp_path, complex_data_dir)
test_resolve_pattern_locally_with_extensions(complex_data_dir, pattern, size, extensions)
test_resolve_pattern_locallyreturns_hidden_dir_only_if_requested(complex_data_dir)

-------------------------methods----------------------
TestDataFilesDict.test_key_order_after_copy(self)


repos/datasets/tests/test_dataset_dict.py
-------------------------functions----------------------
_check_csv_datasetdict(dataset_dict, expected_features, splits = ("train", )
_check_json_datasetdict(dataset_dict, expected_features, splits = ("train", )
_check_parquet_datasetdict(dataset_dict, expected_features, splits = ("train", )
_check_text_datasetdict(dataset_dict, expected_features, splits = ("train", )
test_datasetdict_from_csv_features(features, csv_path, tmp_path)
test_datasetdict_from_csv_keep_in_memory(keep_in_memory, csv_path, tmp_path)
test_datasetdict_from_csv_split(split, csv_path, tmp_path)
test_datasetdict_from_json_features(features, jsonl_path, tmp_path)
test_datasetdict_from_json_keep_in_memory(keep_in_memory, jsonl_path, tmp_path)
test_datasetdict_from_json_splits(split, jsonl_path, tmp_path)
test_datasetdict_from_parquet_features(features, parquet_path, tmp_path)
test_datasetdict_from_parquet_keep_in_memory(keep_in_memory, parquet_path, tmp_path)
test_datasetdict_from_parquet_split(split, parquet_path, tmp_path)
test_datasetdict_from_text_features(features, text_path, tmp_path)
test_datasetdict_from_text_keep_in_memory(keep_in_memory, text_path, tmp_path)
test_datasetdict_from_text_split(split, text_path, tmp_path)
test_dummy_datasetdict_serialize_fs(mockfs)

-------------------------methods----------------------
DatasetDictTest._create_dummy_dataset(self, multiple_columns = False)
DatasetDictTest._create_dummy_dataset_dict(self, multiple_columns = False)
DatasetDictTest._create_dummy_iterable_dataset(self, multiple_columns = False)
DatasetDictTest._create_dummy_iterable_dataset_dict(self, multiple_columns = False)
DatasetDictTest.test_align_labels_with_mapping(self)
DatasetDictTest.test_cast(self)
DatasetDictTest.test_check_values_type(self)
DatasetDictTest.test_filter(self)
DatasetDictTest.test_flatten(self)
DatasetDictTest.test_flatten_indices(self)
DatasetDictTest.test_iterable_filter(self)
DatasetDictTest.test_iterable_map(self)
DatasetDictTest.test_load_from_disk(self)
DatasetDictTest.test_map(self)
DatasetDictTest.test_remove_columns(self)
DatasetDictTest.test_rename_column(self)
DatasetDictTest.test_select_columns(self)
DatasetDictTest.test_serialization(self)
DatasetDictTest.test_set_format_numpy(self)
DatasetDictTest.test_set_format_pandas(self)
DatasetDictTest.test_set_format_polars(self)
DatasetDictTest.test_set_format_tf(self)
DatasetDictTest.test_set_format_torch(self)
DatasetDictTest.test_set_transform(self)
DatasetDictTest.test_shuffle(self)
DatasetDictTest.test_sort(self)
DatasetDictTest.test_with_format(self)
DatasetDictTest.test_with_transform(self)


repos/datasets/tests/test_dataset_list.py
-------------------------methods----------------------
DatasetListTest._create_example_dict(self)
DatasetListTest._create_example_records(self)
DatasetListTest.test_create(self)
DatasetListTest.test_create_empty(self)
DatasetListTest.test_list_dict_equivalent(self)
DatasetListTest.test_uneven_records(self)
DatasetListTest.test_variable_list_records(self)


repos/datasets/tests/test_distributed.py
-------------------------functions----------------------
test_distributed_shuffle_iterable()
test_split_dataset_by_node_iterable()
test_split_dataset_by_node_iterable_sharded(shards_per_node)
test_split_dataset_by_node_map_style()
test_torch_distributed_run(streaming)



repos/datasets/tests/test_download_manager.py
-------------------------functions----------------------
_test_jsonl(path, file)
mock_request(*args, **kwargs)
test_download_manager_delete_extracted_files(xz_file)
test_download_manager_download(urls_type, tmp_path, monkeypatch)
test_download_manager_extract(paths_type, xz_file, text_file, extract_on_the_fly)
test_iter_archive_file(archive_nested_jsonl, request)
test_iter_archive_path(archive_jsonl, request)
test_iter_files(data_dir_with_hidden_files)

-------------------------methods----------------------
MockResponse.iter_content(self, **kwargs)


repos/datasets/tests/test_experimental.py
-------------------------functions----------------------
dummy_function()

-------------------------methods----------------------
TestExperimentalFlag.test_experimental_warning(self)


repos/datasets/tests/test_extract.py
-------------------------functions----------------------
tar_file_with_dot_dot(tmp_path, text_file)
tar_file_with_sym_link(tmp_path)
test_base_extractors(compression_format, is_archive, bz2_file, gz_file, lz4_file, seven_zip_file, tar_file, xz_file, zip_file, zstd_file, tmp_path, text_file, )
test_extractor(compression_format, is_archive, bz2_file, gz_file, lz4_file, seven_zip_file, tar_file, xz_file, zip_file, zstd_file, tmp_path, text_file, )
test_is_zipfile_false_positive(tmpdir)
test_tar_extract_insecure_files(insecure_tar_file, error_log, tar_file_with_dot_dot, tar_file_with_sym_link, tmp_path, caplog)



repos/datasets/tests/test_file_utils.py
-------------------------functions----------------------
_readd_double_slash_removed_by_path(path_as_posix: str)
mock_fsspec2()
test_cached_path_extract(compression_format, gz_file, xz_file, zstd_path, tmp_path, text_file)
test_cached_path_local(text_file)
test_cached_path_missing_local(tmp_path)
test_cached_path_offline()
test_extracted_datasets_path(default_extracted, default_cache_dir, xz_file, tmp_path, monkeypatch)
test_fsspec_offline(tmp_path_factory)
test_ftp_offline(tmp_path_factory)
test_get_extraction_protocol(urlpath, expected_protocol)
test_get_extraction_protocol_gg_drive(urlpath, expected_protocol)
test_get_from_cache_fsspec(tmpfs_file)
test_http_offline(tmp_path_factory)
test_streaming_gg_drive()
test_xdirname(input_path, expected_path)
test_xexists(input_path, exists, tmp_path, mock_fsspec2)
test_xexists_private(hf_private_dataset_repo_txt_data, hf_token)
test_xgetsize(input_path, size, tmp_path, mock_fsspec2)
test_xgetsize_private(hf_private_dataset_repo_txt_data, hf_token)
test_xglob(input_path, expected_paths, tmp_path, mock_fsspec2)
test_xglob_private(hf_private_dataset_repo_zipped_txt_data, hf_token)
test_xisdir(input_path, isdir, tmp_path, mock_fsspec2)
test_xisdir_private(hf_private_dataset_repo_zipped_txt_data, hf_token)
test_xisfile(input_path, isfile, tmp_path, mock_fsspec2)
test_xisfile_private(hf_private_dataset_repo_txt_data, hf_token)
test_xjoin(input_path, paths_to_join, expected_path)
test_xlistdir(input_path, expected_paths, tmp_path, mock_fsspec2)
test_xlistdir_private(hf_private_dataset_repo_zipped_txt_data, hf_token)
test_xnumpy_load(tmp_path)
test_xopen_local(text_path)
test_xopen_remote()
test_xrelpath(input_path, start_path, expected_path)
test_xsplit(input_path, expected_head_and_tail)
test_xsplitext(input_path, expected_path_and_ext)
test_xwalk(input_path, expected_outputs, tmp_path, mock_fsspec2)
test_xwalk_private(hf_private_dataset_repo_zipped_txt_data, hf_token)
tmpfs_file(tmpfs)
zstd_path(tmp_path_factory)

-------------------------methods----------------------
DummyTestFS.__getitem__(self, name)
DummyTestFS._open(self, path, mode = "rb", block_size = None, autocommit = True, cache_options = None, **kwargs, )
DummyTestFS.ls(self, path, detail = True, refresh = True, **kwargs)
TestxPath.test_xpath_as_posix(self, input_path, expected_path)
TestxPath.test_xpath_exists(self, input_path, exists, tmp_path, mock_fsspec2)
TestxPath.test_xpath_glob(self, input_path, pattern, expected_paths, tmp_path, mock_fsspec2)
TestxPath.test_xpath_name(self, input_path, expected)
TestxPath.test_xpath_parent(self, input_path, expected_path)
TestxPath.test_xpath_rglob(self, input_path, pattern, expected_paths, tmp_path, mock_fsspec2)
TestxPath.test_xpath_stem(self, input_path, expected)
TestxPath.test_xpath_str(self, input_path)
TestxPath.test_xpath_suffix(self, input_path, expected)
TestxPath.test_xpath_with_suffix(self, input_path, suffix, expected)


repos/datasets/tests/test_filelock.py
-------------------------functions----------------------
test_long_path(tmpdir)



repos/datasets/tests/test_filesystem.py
-------------------------functions----------------------
test_compression_filesystems(compression_fs_class, gz_file, bz2_file, lz4_file, zstd_file, xz_file, text_file)
test_extract_path_from_uri()
test_fs_isfile(protocol, zip_jsonl_path, jsonl_gz_path)
test_fs_overwrites()
test_is_remote_filesystem(mockfs)
test_mockfs(mockfs)
test_non_mockfs()



repos/datasets/tests/test_fingerprint.py
-------------------------functions----------------------
test_dependency_on_dill()
test_fingerprint_in_multiprocessing()
test_fingerprint_when_transform_version_changes()
test_move_script_doesnt_change_hash(tmp_path: Path)

-------------------------methods----------------------
DatasetChild.func1(self, new_fingerprint, *args, **kwargs)
DatasetChild.func2(self, new_fingerprint, *args, **kwargs)
Foo.__call__(self)
Foo.__init__(self, foo)
HashingTest.test_hash_class_instance(self)
HashingTest.test_hash_same_strings(self)
HashingTest.test_hash_simple(self)
HashingTest.test_hash_spacy_model(self)
HashingTest.test_hash_tiktoken_encoding(self)
HashingTest.test_hash_torch_compiled_function(self)
HashingTest.test_hash_torch_compiled_module(self)
HashingTest.test_hash_torch_generator(self)
HashingTest.test_hash_torch_tensor(self)
HashingTest.test_hash_unpicklable(self)
HashingTest.test_hash_update(self)
HashingTest.test_set_doesnt_depend_on_order(self)
HashingTest.test_set_stable(self)
RecurseHashTest.test_hash_ignores_line_definition_of_function(self)
RecurseHashTest.test_hash_ipython_function(self)
RecurseHashTest.test_recurse_hash_for_class(self)
RecurseHashTest.test_recurse_hash_for_function(self)
RecurseHashTest.test_recurse_hash_for_function_with_shuffled_globals(self)
RecurseHashTest.test_recurse_hash_for_method(self)
TokenizersHashTest.test_hash_regex(self)
TokenizersHashTest.test_hash_tokenizer(self)
TokenizersHashTest.test_hash_tokenizer_with_cache(self)
UnpicklableCallable.__call__(self, *args, **kwargs)
UnpicklableCallable.__getstate__(self)
UnpicklableCallable.__init__(self, callable)


repos/datasets/tests/test_formatting.py
-------------------------functions----------------------
_gen_any_arrays()
any_arrays_dataset()
arrow_table()
test_iterable_dataset_of_arrays_format_to_arrow(any_arrays_dataset: IterableDataset)
test_iterable_dataset_of_arrays_format_to_jax(any_arrays_dataset: IterableDataset)
test_iterable_dataset_of_arrays_format_to_numpy(any_arrays_dataset: IterableDataset)
test_iterable_dataset_of_arrays_format_to_tf(any_arrays_dataset: IterableDataset)
test_iterable_dataset_of_arrays_format_to_torch(any_arrays_dataset: IterableDataset)
test_tf_formatter_sets_default_dtypes(cast_schema, arrow_table)
test_torch_formatter_sets_default_dtypes(cast_schema, arrow_table)

-------------------------methods----------------------
AnyArray.__array__(self)
AnyArray.__init__(self, data)
ArrowExtractorTest._create_dummy_table(self)
ArrowExtractorTest.test_numpy_extractor(self)
ArrowExtractorTest.test_numpy_extractor_nested(self)
ArrowExtractorTest.test_numpy_extractor_temporal(self)
ArrowExtractorTest.test_pandas_extractor(self)
ArrowExtractorTest.test_pandas_extractor_nested(self)
ArrowExtractorTest.test_pandas_extractor_temporal(self)
ArrowExtractorTest.test_polars_extractor(self)
ArrowExtractorTest.test_polars_nested(self)
ArrowExtractorTest.test_polars_temporal(self)
ArrowExtractorTest.test_python_extractor(self)
FormatterTest._create_dummy_table(self)
FormatterTest.test_jax_formatter(self)
FormatterTest.test_jax_formatter_audio(self)
FormatterTest.test_jax_formatter_device(self)
FormatterTest.test_jax_formatter_image(self)
FormatterTest.test_jax_formatter_jnp_array_kwargs(self)
FormatterTest.test_numpy_formatter(self)
FormatterTest.test_numpy_formatter_audio(self)
FormatterTest.test_numpy_formatter_image(self)
FormatterTest.test_numpy_formatter_np_array_kwargs(self)
FormatterTest.test_pandas_formatter(self)
FormatterTest.test_polars_formatter(self)
FormatterTest.test_python_formatter(self)
FormatterTest.test_python_formatter_lazy(self)
FormatterTest.test_tf_formatter(self)
FormatterTest.test_tf_formatter_audio(self)
FormatterTest.test_tf_formatter_image(self)
FormatterTest.test_tf_formatter_tf_tensor_kwargs(self)
FormatterTest.test_torch_formatter(self)
FormatterTest.test_torch_formatter_audio(self)
FormatterTest.test_torch_formatter_image(self)
FormatterTest.test_torch_formatter_torch_tensor_kwargs(self)
LazyDictTest._create_dummy_formatter(self)
LazyDictTest._create_dummy_table(self)
LazyDictTest.test_lazy_dict_copy(self)
QueryTest._create_dummy_arrow_indices(self)
QueryTest._create_dummy_table(self)
QueryTest.assertTableEqual(self, first: pa.Table, second: pa.Table)
QueryTest.test_query_table_indexable_type(self)
QueryTest.test_query_table_int(self)
QueryTest.test_query_table_invalid_key_type(self)
QueryTest.test_query_table_iterable(self)
QueryTest.test_query_table_range(self)
QueryTest.test_query_table_slice(self)
QueryTest.test_query_table_str(self)


repos/datasets/tests/test_hf_gcp.py
-------------------------functions----------------------
list_datasets_on_hf_gcp_parameters(with_config = True, with_revision = True)
test_as_dataset_from_hf_gcs(tmp_path_factory)
test_as_streaming_dataset_from_hf_gcs(tmp_path)

-------------------------methods----------------------
TestDatasetOnHfGcp.test_dataset_info_available(self, dataset, config_name, revision)


repos/datasets/tests/test_hub.py
-------------------------methods----------------------
NewDataset._generate_examples(self)
NewDataset._info(self)
NewDataset._split_generators(self, dl_manager)


repos/datasets/tests/test_info.py
-------------------------functions----------------------
test_dataset_info_dump_and_reload(tmp_path, dataset_info: DatasetInfo)
test_dataset_info_to_yaml_dict()
test_dataset_info_to_yaml_dict_empty()
test_dataset_infos_dict_dump_and_reload(tmp_path, dataset_infos_dict: DatasetInfosDict)
test_from_dir(files, tmp_path_factory)
test_from_merge_same_dataset_infos(dataset_info)



repos/datasets/tests/test_info_utils.py
-------------------------functions----------------------
test_is_small_dataset(dataset_size, input_in_memory_max_size, monkeypatch)



repos/datasets/tests/test_inspect.py
-------------------------functions----------------------
test_get_dataset_config_info(path, config_name, expected_splits)
test_get_dataset_config_info_error(path, config_name, expected_exception)
test_get_dataset_config_info_private(hf_token, hf_private_dataset_repo_txt_data)
test_get_dataset_config_names(path, expected)
test_get_dataset_default_config_name(path, expected)
test_get_dataset_info(path, expected_configs, expected_splits_in_first_config)
test_get_dataset_split_names(path, expected_config, expected_splits)
test_get_dataset_split_names_error(path, config_name, expected_exception)
test_inspect_dataset(path, tmp_path)
test_inspect_metric(path, tmp_path)



repos/datasets/tests/test_iterable_dataset.py
-------------------------functions----------------------
arrow_file(tmp_path_factory, dataset: IterableDataset)
dataset()
dataset_with_several_columns()
filter_func(batch)
generate_examples_fn(**kwargs)
generate_tables_fn(**kwargs)
map_func(batch)
test_arrow_examples_iterable()
test_arrow_examples_iterable_shuffle_data_sources()
test_arrow_examples_iterable_with_kwargs()
test_batch_arrow_tables(tables, batch_size, drop_last_batch)
test_buffer_shuffled_examples_iterable(seed)
test_concatenate_datasets()
test_concatenate_datasets_axis_1()
test_concatenate_datasets_axis_1_resolves_features()
test_concatenate_datasets_axis_1_with_different_lengths()
test_concatenate_datasets_resolves_features()
test_concatenate_datasets_with_different_columns()
test_convert_to_arrow(batch_size, drop_last_batch)
test_cycling_multi_sources_examples_iterable()
test_examples_iterable()
test_examples_iterable_shuffle_data_sources()
test_examples_iterable_shuffle_shards_and_metadata()
test_examples_iterable_with_kwargs()
test_filtered_examples_iterable(n, func, batched, batch_size)
test_filtered_examples_iterable_input_columns(n, func, batched, batch_size, input_columns)
test_filtered_examples_iterable_with_indices(n, func, batched, batch_size)
test_formatted_map(dataset: IterableDataset)
test_from_spark_streaming()
test_from_spark_streaming_features()
test_horizontally_concatenated_examples_iterable()
test_interleave_dataset_with_sharding(n_shards1, n_shards2, num_workers)
test_interleave_datasets(dataset: IterableDataset, probas, seed, expected_length, stopping_strategy)
test_interleave_datasets_with_features(dataset: IterableDataset, )
test_interleave_datasets_with_oversampling()
test_iter_arrow(ex_iterable: _BaseExamplesIterable)
test_iterable_dataset()
test_iterable_dataset_add_column(dataset_with_several_columns)
test_iterable_dataset_cast()
test_iterable_dataset_cast_column()
test_iterable_dataset_features(features)
test_iterable_dataset_features_cast_to_python()
test_iterable_dataset_filter(dataset: IterableDataset)
test_iterable_dataset_from_file(dataset: IterableDataset, arrow_file: str)
test_iterable_dataset_from_generator()
test_iterable_dataset_from_generator_with_shards()
test_iterable_dataset_from_hub_torch_dataloader_parallel(num_workers, tmp_path)
test_iterable_dataset_info()
test_iterable_dataset_is_torch_iterable_dataset(dataset: IterableDataset)
test_iterable_dataset_iter_batch(batch_size, drop_last_batch)
test_iterable_dataset_map(dataset: IterableDataset, )
test_iterable_dataset_map_batched(dataset: IterableDataset, )
test_iterable_dataset_map_complex_features(dataset: IterableDataset, )
test_iterable_dataset_map_with_features(dataset: IterableDataset)
test_iterable_dataset_map_with_fn_kwargs(dataset: IterableDataset)
test_iterable_dataset_remove_columns(dataset_with_several_columns)
test_iterable_dataset_rename_column(dataset_with_several_columns)
test_iterable_dataset_rename_columns(dataset_with_several_columns)
test_iterable_dataset_resolve_features()
test_iterable_dataset_resolve_features_keep_order()
test_iterable_dataset_select_columns(dataset_with_several_columns)
test_iterable_dataset_set_epoch(dataset: IterableDataset)
test_iterable_dataset_set_epoch_of_shuffled_dataset(dataset: IterableDataset, seed, epoch)
test_iterable_dataset_shuffle(dataset: IterableDataset, seed, epoch)
test_iterable_dataset_shuffle_after_skip_or_take(method)
test_iterable_dataset_skip(dataset: IterableDataset, n)
test_iterable_dataset_take(dataset: IterableDataset, n)
test_iterable_dataset_torch_dataloader_parallel()
test_iterable_dataset_torch_integration()
test_iterable_dataset_torch_picklable()
test_iterable_dataset_with_features_fill_with_none()
test_iterable_dataset_with_format(dataset: IterableDataset, format_type)
test_iterable_dataset_with_format_torch()
test_map_array_are_not_converted_back_to_lists(dataset: IterableDataset)
test_mapped_examples_iterable(n, func, batched, batch_size)
test_mapped_examples_iterable_arrow_format(n, func, batched, batch_size)
test_mapped_examples_iterable_drop_last_batch(n, func, batched, batch_size)
test_mapped_examples_iterable_drop_last_batch_and_arrow_format(n, func, batched, batch_size)
test_mapped_examples_iterable_fn_kwargs(n, func, batched, batch_size, fn_kwargs)
test_mapped_examples_iterable_fn_kwargs_and_arrow_format(n, func, batched, batch_size, fn_kwargs)
test_mapped_examples_iterable_input_columns(n, func, batched, batch_size, input_columns)
test_mapped_examples_iterable_input_columns_and_arrow_format(n, func, batched, batch_size, input_columns)
test_mapped_examples_iterable_remove_columns(n, func, batched, batch_size, remove_columns)
test_mapped_examples_iterable_remove_columns_arrow_format(n, func, batched, batch_size, remove_columns)
test_mapped_examples_iterable_with_indices(n, func, batched, batch_size)
test_mapped_examples_iterable_with_indices_and_arrow_format(n, func, batched, batch_size)
test_no_iter_arrow(ex_iterable: _BaseExamplesIterable)
test_pickle_after_many_transforms(dataset_with_several_columns)
test_randomly_cycling_multi_sources_examples_iterable(probabilities)
test_sharded_iterable_dataset_torch_dataloader_parallel(n_shards, num_workers)
test_skip_examples_iterable()
test_take_examples_iterable()
test_vertically_concatenated_examples_iterable()
test_vertically_concatenated_examples_iterable_shuffle_data_sources()
test_vertically_concatenated_examples_iterable_with_different_columns()
test_with_format_tf(dataset_with_several_columns: IterableDataset)
test_with_format_torch(dataset_with_several_columns: IterableDataset)



repos/datasets/tests/test_load.py
-------------------------functions----------------------
data_dir(tmp_path)
data_dir_with_arrow(tmp_path)
data_dir_with_metadata(tmp_path)
data_dir_with_single_config_in_metadata(tmp_path)
data_dir_with_two_config_in_metadata(tmp_path)

-------------------------methods----------------------
__DummyDataset1__._generate_examples(self, filepath, **kwargs)
__DummyDataset1__._info(self)
__DummyDataset1__._split_generators(self, dl_manager)
__DummyMetric1__._compute(self, predictions, references)
__DummyMetric1__._info(self)


repos/datasets/tests/test_metadata_util.py
-------------------------functions----------------------
_dedent(string: str)
data_dir_with_two_subdirs(tmp_path)
test_metadata_configs_dataset_card_data(readme_content, expected_metadata_configs_dict, expected_default_config_name)
test_metadata_configs_incorrect_yaml()
test_split_order_in_metadata_configs_from_exported_parquet_files_and_dataset_infos()

-------------------------methods----------------------
TestMetadataUtils.test_from_yaml_string(self)
TestMetadataUtils.test_metadata_dict_from_readme(self)


repos/datasets/tests/test_metric.py
-------------------------functions----------------------
metric_add_and_compute(arg)
metric_add_batch_and_compute(arg)
metric_compute(arg)
properly_del_metric(metric)
test_metric_with_multilabel(config_name, predictions, references, expected, tmp_path)
test_metric_with_non_standard_feature_names_add(tmp_path)
test_metric_with_non_standard_feature_names_add_batch(tmp_path)
test_metric_with_non_standard_feature_names_compute(tmp_path)
test_safety_checks_process_vars()

-------------------------methods----------------------
AccuracyWithNonStandardFeatureNames._compute(self, inputs, targets)
AccuracyWithNonStandardFeatureNames._info(self)
AccuracyWithNonStandardFeatureNames.expected_results(cls)
AccuracyWithNonStandardFeatureNames.inputs_and_targets(cls)
DummyMetric._compute(self, predictions, references)
DummyMetric._info(self)
DummyMetric.distributed_expected_results(cls)
DummyMetric.distributed_predictions_and_references(cls)
DummyMetric.expected_results(cls)
DummyMetric.other_expected_results(cls)
DummyMetric.other_predictions_and_references(cls)
DummyMetric.predictions_and_references(cls)
DummyMetric.separate_expected_results(cls)
DummyMetric.separate_predictions_and_references(cls)
MetricWithMultiLabel._compute(self, predictions = None, references = None)
MetricWithMultiLabel._info(self)
TestMetric.test_concurrent_metrics(self)
TestMetric.test_distributed_metrics(self)
TestMetric.test_dummy_metric(self)
TestMetric.test_dummy_metric_pickle(self)
TestMetric.test_input_numpy(self)
TestMetric.test_input_tf(self)
TestMetric.test_input_torch(self)
TestMetric.test_metric_with_cache_dir(self)
TestMetric.test_separate_experiments_in_parallel(self)


repos/datasets/tests/test_metric_common.py
-------------------------functions----------------------
get_local_metric_names()
patch_bertscore(module_name)
patch_bleurt(module_name)
patch_comet(module_name)
skip_if_metric_requires_fairseq(test_case)
skip_if_metric_requires_transformers(test_case)
skip_on_windows_if_not_windows_compatible(test_case)
test_seqeval_raises_when_incorrect_scheme()

-------------------------methods----------------------
LocalMetricTest.patch_intensive_calls(self, metric_name, module_name)
LocalMetricTest.register_intensive_calls_patcher(cls, metric_name)
LocalMetricTest.test_load_metric(self, metric_name)
LocalMetricTest.test_load_real_metric(self, metric_name)
LocalMetricTest.use_local_metrics(self)


repos/datasets/tests/test_offline_util.py
-------------------------functions----------------------
test_offline_with_connection_error()
test_offline_with_datasets_offline_mode_enabled()
test_offline_with_timeout()



repos/datasets/tests/test_parallel.py
-------------------------functions----------------------
add_one(i)
test_parallel_backend_input()
test_parallel_backend_map_nested(num_proc)



repos/datasets/tests/test_patching.py
-------------------------functions----------------------
test_patch_submodule()
test_patch_submodule_builtin()
test_patch_submodule_doesnt_exist()
test_patch_submodule_missing()
test_patch_submodule_missing_builtin()
test_patch_submodule_start_and_stop()
test_patch_submodule_successive()



repos/datasets/tests/test_py_utils.py
-------------------------functions----------------------
_2seconds_generator_of_2items_with_timing(content)
_split_text(text: str)
add_one(i)
add_one_to_batch(batch)
np_sum(x)
test_asdict()
test_flatten(data, expected_output)
test_iflatmap_unordered()
test_map_nested(data_struct, expected_result, num_proc, batched, function)
test_map_nested_num_proc(iterable_length, num_proc, expected_num_proc)
test_nested_data_structure_data(input_data)

-------------------------methods----------------------
PyUtilsTest.test_map_nested(self)
PyUtilsTest.test_temporary_assignment(self)
PyUtilsTest.test_zip_dict(self)
TempSeedTest.test_numpy(self)
TempSeedTest.test_tensorflow(self)
TempSeedTest.test_torch(self)


repos/datasets/tests/test_readme_util.py
-------------------------functions----------------------
test_readme_from_readme_correct(readme_md, expected_dict)
test_readme_from_readme_error(readme_md, expected_error)
test_readme_from_readme_parsing_errors(readme_md, expected_error)
test_readme_from_readme_suppress_parsing_errors(readme_md)
test_readme_from_string_correct(readme_md, expected_dict)
test_readme_from_string_parsing_errors(readme_md, expected_error)
test_readme_from_string_suppress_parsing_errors(readme_md)
test_readme_from_string_validation_errors(readme_md, expected_error)



repos/datasets/tests/test_search.py
-------------------------functions----------------------
test_serialization_fs(mockfs)

-------------------------methods----------------------
ElasticSearchIndexTest.test_elasticsearch(self)
FaissIndexTest.test_custom(self)
FaissIndexTest.test_factory(self)
FaissIndexTest.test_flat_ip(self)
FaissIndexTest.test_serialization(self)
IndexableDatasetTest._create_dummy_dataset(self)
IndexableDatasetTest.test_add_elasticsearch_index(self)
IndexableDatasetTest.test_add_faiss_index(self)
IndexableDatasetTest.test_add_faiss_index_errors(self)
IndexableDatasetTest.test_add_faiss_index_from_external_arrays(self)
IndexableDatasetTest.test_drop_index(self)
IndexableDatasetTest.test_serialization(self)


repos/datasets/tests/test_sharding_utils.py
-------------------------functions----------------------
test_distribute_shards(kwargs, expected)
test_number_of_shards_in_gen_kwargs(gen_kwargs, expected)
test_split_gen_kwargs(gen_kwargs, max_num_jobs, expected)



repos/datasets/tests/test_splits.py
-------------------------functions----------------------
test_split_dict_asdict_has_dataset_name(split_info)
test_split_dict_to_yaml_list(split_dict: SplitDict)



repos/datasets/tests/test_streaming_download_manager.py
-------------------------functions----------------------
_test_jsonl(path, file)
test_iter_archive_file(archive_nested_jsonl, request)
test_iter_archive_path(archive_jsonl, request)
test_iter_files(data_dir_with_hidden_files)
test_streaming_dl_manager_download(text_path)
test_streaming_dl_manager_download_and_extract_no_extraction(urlpath)
test_streaming_dl_manager_download_and_extract_with_extraction(text_gz_path, text_path)
test_streaming_dl_manager_download_and_extract_with_join(input_path, filename, expected_path)
test_streaming_dl_manager_download_dummy_path(urlpath)
test_streaming_dl_manager_extract(text_gz_path, text_path)
test_streaming_dl_manager_extract_all_supported_single_file_compression_types(compression_fs_class, gz_file, xz_file, zstd_file, bz2_file, lz4_file, text_file)
test_streaming_dl_manager_extract_throws(urlpath)
test_streaming_gg_drive_gzipped()
test_streaming_gg_drive_no_extract()
test_streaming_gg_drive_zipped()



repos/datasets/tests/test_table.py
-------------------------functions----------------------
_interpolation_search_ground_truth(arr: List[int], x: int)
_to_testing_blocks(table: TableBlock)
add_suffix_to_column_names(table, suffix)
assert_deepcopy_does_bring_data_in_memory(table: MemoryMappedTable)
assert_deepcopy_without_bringing_data_in_memory(table: MemoryMappedTable)
assert_index_attributes_equal(table: Table, other: Table)
assert_pickle_does_bring_data_in_memory(table: MemoryMappedTable)
assert_pickle_without_bringing_data_in_memory(table: MemoryMappedTable)
in_memory_blocks(in_memory_pa_table)
in_memory_pa_table(arrow_file)
memory_mapped_blocks(arrow_file)
mixed_in_memory_and_memory_mapped_blocks(in_memory_blocks, memory_mapped_blocks)
test_cast_array_to_features_array_xd()
test_cast_array_to_features_nested()
test_cast_array_to_features_nested_with_nulls()
test_cast_array_to_features_sequence_classlabel()
test_cast_array_to_features_to_nested_with_no_fields()
test_cast_array_to_features_to_null_type()
test_cast_array_xd_to_features_sequence()
test_cast_boolean_array_to_features()
test_cast_decimal_array_to_features()
test_cast_fixed_size_list_array_to_features_sequence(arr, slice, target_value_feature)
test_cast_float_array_to_features()
test_cast_integer_array_to_features()
test_cast_list_array_to_features_sequence(arr, slice, target_value_feature)
test_concat_tables(arrow_file, in_memory_pa_table)
test_concat_tables_cast_with_features_metadata(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concat_tables_with_features_metadata(arrow_file, in_memory_pa_table)
test_concatenation_table_add_column(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_append_column(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_cast(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_combine_chunks(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_deepcopy(blocks_type, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_drop(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_filter(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_flatten(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_from_blocks(in_memory_pa_table, in_memory_blocks)
test_concatenation_table_from_blocks_doesnt_increase_memory(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_from_tables(axis, in_memory_pa_table, arrow_file)
test_concatenation_table_from_tables_axis1_misaligned_blocks(arrow_file)
test_concatenation_table_init(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_pickle(blocks_type, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_remove_column(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_rename_columns(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_replace_schema_metadata(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_set_column(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_slice(blocks_type, in_memory_pa_table, in_memory_blocks, memory_mapped_blocks, mixed_in_memory_and_memory_mapped_blocks)
test_concatenation_table_slice_mixed_schemas_vertically(arrow_file)
test_embed_array_storage(image_file)
test_embed_array_storage_nested(image_file)
test_embed_table_storage(image_file)
test_in_memory_arrow_table_from_buffer(in_memory_pa_table)
test_in_memory_arrow_table_from_file(arrow_file, in_memory_pa_table)
test_in_memory_table_add_column(in_memory_pa_table)
test_in_memory_table_append_column(in_memory_pa_table)
test_in_memory_table_cast(in_memory_pa_table)
test_in_memory_table_cast_reorder_struct()
test_in_memory_table_cast_with_hf_features()
test_in_memory_table_combine_chunks(in_memory_pa_table)
test_in_memory_table_deepcopy(in_memory_pa_table)
test_in_memory_table_drop(in_memory_pa_table)
test_in_memory_table_filter(in_memory_pa_table)
test_in_memory_table_flatten(in_memory_pa_table)
test_in_memory_table_from_arrays(in_memory_pa_table)
test_in_memory_table_from_batches(in_memory_pa_table)
test_in_memory_table_from_buffer(in_memory_pa_table)
test_in_memory_table_from_file(arrow_file, in_memory_pa_table)
test_in_memory_table_from_pandas(in_memory_pa_table)
test_in_memory_table_from_pydict(in_memory_pa_table)
test_in_memory_table_from_pylist(in_memory_pa_table)
test_in_memory_table_pickle(in_memory_pa_table)
test_in_memory_table_pickle_big_table()
test_in_memory_table_remove_column(in_memory_pa_table)
test_in_memory_table_rename_columns(in_memory_pa_table)
test_in_memory_table_replace_schema_metadata(in_memory_pa_table)
test_in_memory_table_set_column(in_memory_pa_table)
test_in_memory_table_slice(in_memory_pa_table)
test_indexed_table_mixin()
test_inject_arrow_table_documentation(in_memory_pa_table)
test_interpolation_search(arr, x)
test_memory_mapped_arrow_table_from_file(arrow_file, in_memory_pa_table)
test_memory_mapped_table_add_column(arrow_file, in_memory_pa_table)
test_memory_mapped_table_append_column(arrow_file, in_memory_pa_table)
test_memory_mapped_table_cast(arrow_file, in_memory_pa_table)
test_memory_mapped_table_combine_chunks(arrow_file, in_memory_pa_table)
test_memory_mapped_table_deepcopy(arrow_file)
test_memory_mapped_table_drop(arrow_file, in_memory_pa_table)
test_memory_mapped_table_filter(arrow_file, in_memory_pa_table)
test_memory_mapped_table_flatten(arrow_file, in_memory_pa_table)
test_memory_mapped_table_from_file(arrow_file, in_memory_pa_table)
test_memory_mapped_table_from_file_with_replay(arrow_file, in_memory_pa_table)
test_memory_mapped_table_init(arrow_file, in_memory_pa_table)
test_memory_mapped_table_pickle(arrow_file)
test_memory_mapped_table_pickle_applies_replay(arrow_file)
test_memory_mapped_table_pickle_doesnt_fill_memory(arrow_file)
test_memory_mapped_table_remove_column(arrow_file, in_memory_pa_table)
test_memory_mapped_table_rename_columns(arrow_file, in_memory_pa_table)
test_memory_mapped_table_replace_schema_metadata(arrow_file, in_memory_pa_table)
test_memory_mapped_table_set_column(arrow_file, in_memory_pa_table)
test_memory_mapped_table_slice(arrow_file, in_memory_pa_table)
test_table_attributes(in_memory_pa_table, attribute)
test_table_column(in_memory_pa_table)
test_table_equals(in_memory_pa_table)
test_table_field(in_memory_pa_table)
test_table_getitem(in_memory_pa_table)
test_table_init(in_memory_pa_table)
test_table_iter(table, batch_size, drop_last_batch)
test_table_itercolumns(in_memory_pa_table)
test_table_len(in_memory_pa_table)
test_table_str(in_memory_pa_table)
test_table_to_batches(in_memory_pa_table)
test_table_to_pydict(in_memory_pa_table)
test_table_to_string(in_memory_pa_table)
test_table_validate(in_memory_pa_table)

-------------------------methods----------------------
_ListWithGetitemCounter.__getitem__(self, i)
_ListWithGetitemCounter.__init__(self, *args, **kwargs)
_ListWithGetitemCounter.getitem_unique_count(self)


repos/datasets/tests/test_tasks.py
-------------------------functions----------------------
test_reload_task_from_dict(task_cls)

-------------------------methods----------------------
AudioClassificationTest.setUp(self)
AudioClassificationTest.test_align_with_features(self)
AudioClassificationTest.test_column_mapping(self)
AudioClassificationTest.test_from_dict(self)
AutomaticSpeechRecognitionTest.test_column_mapping(self)
AutomaticSpeechRecognitionTest.test_from_dict(self)
DatasetWithTaskProcessingTest.test_map_on_task_template(self)
DatasetWithTaskProcessingTest.test_remove_and_map_on_task_template(self)
ImageClassificationTest.setUp(self)
ImageClassificationTest.test_align_with_features(self)
ImageClassificationTest.test_column_mapping(self)
ImageClassificationTest.test_from_dict(self)
QuestionAnsweringTest.test_column_mapping(self)
QuestionAnsweringTest.test_from_dict(self)
SummarizationTest.test_column_mapping(self)
SummarizationTest.test_from_dict(self)
TestLanguageModeling.test_column_mapping(self)
TestLanguageModeling.test_from_dict(self)
TextClassificationTest.setUp(self)
TextClassificationTest.test_align_with_features(self)
TextClassificationTest.test_column_mapping(self)
TextClassificationTest.test_from_dict(self)


repos/datasets/tests/test_tqdm.py
-------------------------methods----------------------
TestTqdmUtils.capsys(self, capsys: CaptureFixture)
TestTqdmUtils.setUp(self)
TestTqdmUtils.tearDown(self)
TestTqdmUtils.test_cannot_disable_tqdm_when_env_variable_is_set(self)
TestTqdmUtils.test_cannot_enable_tqdm_when_env_variable_is_set(self)
TestTqdmUtils.test_tqdm_can_be_disabled_when_globally_enabled(self)
TestTqdmUtils.test_tqdm_disabled(self)
TestTqdmUtils.test_tqdm_disabled_cannot_be_forced(self)
TestTqdmUtils.test_tqdm_enabled(self)
TestTqdmUtils.test_tqdm_helpers(self)


repos/datasets/tests/test_upstream_hub.py
-------------------------functions----------------------
text_file_with_metadata(request, tmp_path, text_file)

-------------------------methods----------------------
TestPushToHub.test_push_dataset_dict_to_hub(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_custom_features(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_custom_splits(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_datasets_with_different_features(self, cleanup_repo)
TestPushToHub.test_push_dataset_dict_to_hub_multiple_files(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_multiple_files_with_max_shard_size(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_multiple_files_with_num_shards(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_name_without_namespace(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_no_token(self, temporary_repo, set_ci_hub_access_token)
TestPushToHub.test_push_dataset_dict_to_hub_overwrite_files(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_private(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_with_config_no_metadata_configs(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_with_multiple_commits(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_with_pull_request(self, temporary_repo)
TestPushToHub.test_push_dataset_dict_to_hub_with_revision(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub_custom_features(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub_custom_features_audio(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub_custom_features_image(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub_custom_features_image_list(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub_custom_splits(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub_multiple_splits_one_by_one(self, temporary_repo)
TestPushToHub.test_push_dataset_to_hub_with_config_no_metadata_configs(self, temporary_repo)
TestPushToHub.test_push_multiple_dataset_configs_to_hub_load_dataset(self, temporary_repo)
TestPushToHub.test_push_multiple_dataset_configs_to_hub_load_dataset_builder(self, temporary_repo)
TestPushToHub.test_push_multiple_dataset_configs_to_hub_readme_metadata_content(self, specific_default_config_name, temporary_repo)
TestPushToHub.test_push_multiple_dataset_dict_configs_to_hub_load_dataset(self, temporary_repo)
TestPushToHub.test_push_multiple_dataset_dict_configs_to_hub_load_dataset_builder(self, temporary_repo)
TestPushToHub.test_push_multiple_dataset_dict_configs_to_hub_readme_metadata_content(self, specific_default_config_name, temporary_repo)
TestPushToHub.test_push_streaming_dataset_dict_to_hub(self, temporary_repo)


repos/datasets/tests/test_version.py
-------------------------functions----------------------
test_version_equality_and_hash(other, expected_equality)



repos/datasets/tests/test_warnings.py
-------------------------functions----------------------
mock_emitted_deprecation_warnings(monkeypatch)
mock_hfh(monkeypatch)
test_metric_deprecation_warning(func, args, mock_emitted_deprecation_warnings, mock_hfh, tmp_path)



repos/datasets/tests/utils.py
-------------------------functions----------------------
assert_arrow_memory_doesnt_increase()
assert_arrow_memory_increases()
execute_subprocess_async(cmd, env = None, stdin = None, timeout = 180, quiet = False, echo = True)
for_all_test_methods(*decorators)
get_torch_dist_unique_port()
is_rng_equal(rng1, rng2)
local(test_case)
offline(mode = OfflineSimulationMode.CONNECTION_FAILS, timeout = 1e-16)
packaged(test_case)
parse_flag_from_env(key, default = False)
pytest_xdist_worker_id()
remote(test_case)
require_elasticsearch(test_case)
require_faiss(test_case)
require_jax(test_case)
require_joblibspark(test_case)
require_pil(test_case)
require_polars(test_case)
require_pyspark(test_case)
require_regex(test_case)
require_spacy(test_case)
require_sqlalchemy(test_case)
require_tf(test_case)
require_tiktoken(test_case)
require_torch(test_case)
require_transformers(test_case)
set_current_working_directory_to_temp_dir(*args, **kwargs)
slow(test_case)
xfail_if_500_502_http_error(func)

-------------------------methods----------------------
_RunOutput.__init__(self, returncode, stdout, stderr)


repos/datasets/utils/release.py
-------------------------functions----------------------
get_version()
global_version_update(version)
post_release_work()
pre_release_work(patch = False)
update_version_in_file(fname, version, pattern)



repos/llama_index/_llama-index/llama_index/_bundle/__init__.py


repos/llama_index/benchmarks/agent/agent_utils.py
-------------------------functions----------------------
get_model(model: str)
is_valid_combination(agent: str, model: str)



repos/llama_index/benchmarks/agent/button_tasks.py
-------------------------functions----------------------
get_dial_then_enter()
get_search_then_dial()
search_number(first_name: str, last_name: str)

-------------------------methods----------------------
Phone.__init__(self)
Phone.dial_digit(self, number: str)
Phone.enter(self)
Phone.evaluate(self, response: str, expected_response: str)


repos/llama_index/benchmarks/agent/eval.py
-------------------------functions----------------------
contains_expected_response(response: str, expected_response: str)



repos/llama_index/benchmarks/agent/main.py
-------------------------functions----------------------
benchmark(AGENTS.keys()), models: List[str]  =  ALL_MODELS, tasks: List[str]  =  ALL_TASKS, verbose: bool  =  False, output: str  =  "results.csv", save: bool  =  True, )
evaluate(agent: str, model: str, task_name: str, verbose: bool  =  False)



repos/llama_index/benchmarks/agent/math_tasks.py
-------------------------functions----------------------
add(a: int, b: int)
multiply(a: int, b: int)



repos/llama_index/benchmarks/agent/task.py


repos/llama_index/benchmarks/embeddings/bench_embeddings.py
-------------------------functions----------------------
generate_strings(num_strings: int  =  100, string_length: int  =  10)



repos/llama_index/benchmarks/struct_indices/spider/evaluate.py
-------------------------functions----------------------
_answer(llm: OpenAI, question: str, sql_query: str, sql_result: Optional[str])
_get_answers(llm: OpenAI, indexes: Dict[str, SQLStructStoreIndex], db_names: List[str], sql_queries: List[str], examples: List[dict], output_filename: str, use_cache: bool, )
_match(llm: OpenAI, question: str, reference_answer: str, hypothesis_answer: str)
_match_answers(llm: OpenAI, gold_results: List[dict], pred_results: List[dict], examples: List[dict], output_filename: str, )



repos/llama_index/benchmarks/struct_indices/spider/generate_sql.py
-------------------------functions----------------------
_generate_sql(llama_index: SQLStructStoreIndex, nl_query_text: str, )
generate_sql(llama_indexes: dict, examples: list, output_file: str)



repos/llama_index/benchmarks/struct_indices/spider/sample_benchmark.py


repos/llama_index/benchmarks/struct_indices/spider/spider_utils.py
-------------------------functions----------------------
create_indexes(spider_dir: str, llm: OpenAI)
load_examples(spider_dir: str)



repos/llama_index/benchmarks/vector_stores/bench_simple_vector_store.py
-------------------------functions----------------------
bench_simple_vector_store(num_vectors: List[int]  =  [10, 50, 100, 500, 1000])
generate_nodes(num_vectors: int  =  100, embedding_length: int  =  1536)



repos/llama_index/docs/prepare_for_build.py


repos/llama_index/experimental/classifier/utils.py
-------------------------functions----------------------
extract_float_given_response(response: str, n: int  =  1)
get_eval_preds(train_prompt: BasePromptTemplate, train_str: str, eval_df: pd.DataFrame, n: int  =  20)
get_label_str(labels: pd.Series, i: int)
get_sorted_dict_str(d: dict)
get_train_and_eval_data(csv_path: str, )
get_train_str(train_df: pd.DataFrame, train_labels: pd.Series, train_n: int  =  10)



repos/llama_index/experimental/cli/__init__.py


repos/llama_index/experimental/cli/__main__.py
-------------------------functions----------------------
main()



repos/llama_index/experimental/cli/cli_add.py
-------------------------functions----------------------
add_cli(args: Namespace)
register_add_cli(subparsers: _SubParsersAction)



repos/llama_index/experimental/cli/cli_init.py
-------------------------functions----------------------
init_cli(args: Namespace)
register_init_cli(subparsers: _SubParsersAction)



repos/llama_index/experimental/cli/cli_query.py
-------------------------functions----------------------
query_cli(args: Namespace)
register_query_cli(subparsers: _SubParsersAction)



repos/llama_index/experimental/cli/configuration.py
-------------------------functions----------------------
_load_embed_model(config: ConfigParser)
_load_llm(section: SectionProxy)
_load_llm_predictor(config: ConfigParser)
_load_service_context(config: ConfigParser)
_load_storage_context(config: ConfigParser)
load_config(root: str  =  ".")
load_index(root: str  =  ".")
save_config(config: ConfigParser, root: str  =  ".")
save_index(index: BaseIndex[Any], root: str  =  ".")



repos/llama_index/experimental/openai_fine_tuning/launch_training.py
-------------------------functions----------------------
launch_training(data_path: str)



repos/llama_index/experimental/openai_fine_tuning/validate_json.py
-------------------------functions----------------------
validate_json(data_path: str)



repos/llama_index/experimental/splitter_playground/app.py
-------------------------functions----------------------
load_document(uploaded_files: List[UploadedFile])



repos/llama_index/llama-datasets/10k/uber_2021/llamaindex_baseline.py


repos/llama_index/llama-datasets/__init__.py


repos/llama_index/llama-datasets/blockchain_solana/llamaindex_baseline.py


repos/llama_index/llama-datasets/braintrust_coda/__init__.py


repos/llama_index/llama-datasets/braintrust_coda/llamaindex_baseline.py


repos/llama_index/llama-datasets/covidqa/llamaindex_baseline.py


repos/llama_index/llama-datasets/docugami_kg_rag/sec_10_q/llamaindex_baseline.py


repos/llama_index/llama-datasets/eval_llm_survey_paper/llamaindex_baseline.py


repos/llama_index/llama-datasets/history_of_alexnet/llamaindex_baseline.py


repos/llama_index/llama-datasets/llama2_paper/__init__.py


repos/llama_index/llama-datasets/llama2_paper/llamaindex_baseline.py


repos/llama_index/llama-datasets/mini_covidqa/llamaindex_baseline.py


repos/llama_index/llama-datasets/mini_esg_bench/llamaindex_baseline.py


repos/llama_index/llama-datasets/mini_mt_bench_singlegrading/baselines.py


repos/llama_index/llama-datasets/mini_squadv2/llamaindex_baseline.py


repos/llama_index/llama-datasets/mini_truthfulqa/llamaindex_baseline.py


repos/llama_index/llama-datasets/mt_bench_humanjudgement/baselines.py


repos/llama_index/llama-datasets/origin_of_covid19/llamaindex_baseline.py


repos/llama_index/llama-datasets/patronus_financebench/__init__.py


repos/llama_index/llama-datasets/patronus_financebench/llamaindex_baseline.py


repos/llama_index/llama-datasets/paul_graham_essay/__init__.py


repos/llama_index/llama-datasets/paul_graham_essay/llamaindex_baseline.py


repos/llama_index/llama-index-cli/llama_index/cli/__init__.py


repos/llama_index/llama-index-cli/llama_index/cli/command_line.py
-------------------------functions----------------------
default_rag_cli()
handle_download_llama_dataset(llama_dataset_class: Optional[str]  =  None, download_dir: Optional[str]  =  None, llama_hub_url: str  =  LLAMA_HUB_URL, llama_datasets_lfs_url: str  =  LLAMA_DATASETS_LFS_URL, llama_datasets_source_files_tree_url: str  =  LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL, **kwargs: Any, )
handle_download_llama_pack(llama_pack_class: Optional[str]  =  None, download_dir: Optional[str]  =  None, llama_pack_url: str  =  LLAMA_PACKS_CONTENTS_URL, **kwargs: Any, )
handle_init_package(name: str, kind: str, prefix: Optional[str]  =  None, **kwargs: Any)
main()



repos/llama_index/llama-index-cli/tests/__init__.py


repos/llama_index/llama-index-cli/tests/test_cli.py


repos/llama_index/llama-index-core/llama_index/core/__init__.py


repos/llama_index/llama-index-core/llama_index/core/async_utils.py
-------------------------functions----------------------
asyncio_module(show_progress: bool  =  False)
asyncio_run(coro: Coroutine)
chunks(iterable: Iterable, size: int)
get_asyncio_module(show_progress: bool  =  False)
run_async_tasks(tasks: List[Coroutine], show_progress: bool  =  False, progress_bar_desc: str  =  "Running async tasks", )



repos/llama_index/llama-index-core/llama_index/core/constants.py


repos/llama_index/llama-index-core/llama_index/core/exec_utils.py
-------------------------functions----------------------
_contains_protected_access(code: str)
_get_restricted_globals(__globals: Union[dict, None])
_restricted_import(name: str, globals: Union[Mapping[str, object], None]  =  None, locals: Union[Mapping[str, object], None]  =  None, ), level: int  =  0, )
_verify_source_safety(__source: Union[str, bytes, CodeType])
safe_eval(__source: Union[str, bytes, CodeType], __globals: Union[Dict[str, Any], None]  =  None, __locals: Union[Mapping[str, object], None]  =  None, )
safe_exec(__source: Union[str, bytes, CodeType], __globals: Union[Dict[str, Any], None]  =  None, __locals: Union[Mapping[str, object], None]  =  None, )

-------------------------methods----------------------
DunderVisitor.__init__(self)
DunderVisitor.visit_Attribute(self, node: ast.Attribute)
DunderVisitor.visit_Name(self, node: ast.Name)


repos/llama_index/llama-index-core/llama_index/core/image_retriever.py
-------------------------methods----------------------
BaseImageRetriever._image_to_image_retrieve(self, query_bundle: QueryBundle, )
BaseImageRetriever._text_to_image_retrieve(self, query_bundle: QueryBundle, )
BaseImageRetriever.image_to_image_retrieve(self, str_or_query_bundle: QueryType)
BaseImageRetriever.text_to_image_retrieve(self, str_or_query_bundle: QueryType)


repos/llama_index/llama-index-core/llama_index/core/img_utils.py
-------------------------functions----------------------
b64_2_img(data: str)
img_2_b64(image: Image, format: str  =  "JPEG")



repos/llama_index/llama-index-core/llama_index/core/schema.py
-------------------------methods----------------------
BaseComponent.__getstate__(self)
BaseComponent.__setstate__(self, state: Dict[str, Any])
BaseComponent.class_name(cls)
BaseComponent.dict(self, **kwargs: Any)
BaseComponent.from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignorekwargs, dict))
BaseComponent.from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignoredata_str)data, **kwargs))
BaseComponent.json(self, **kwargs: Any)
BaseComponent.to_dict(self, **kwargs: Any)
BaseComponent.to_json(self, **kwargs: Any)
BaseNode.__str__(self)
BaseNode.as_related_node_info(self)
BaseNode.child_nodes(self)
BaseNode.extra_info(self)
BaseNode.get_content(self, metadata_mode: MetadataMode  =  MetadataMode.ALL)
BaseNode.get_embedding(self)
BaseNode.get_metadata_str(self, mode: MetadataMode  =  MetadataMode.ALL)
BaseNode.get_type(cls)
BaseNode.hash(self)
BaseNode.next_node(self)
BaseNode.node_id(self)
BaseNode.node_id(self)
BaseNode.parent_node(self)
BaseNode.prev_node(self)
BaseNode.ref_doc_id(self)
BaseNode.set_content(self, value: Any)
BaseNode.source_node(self)
Document.__setattr__(self, name: str, value: object)
Document.__str__(self)
Document.class_name(cls)
Document.doc_id(self)
Document.example(cls)
Document.from_embedchain_format(cls, doc: Dict[str, Any])
Document.from_haystack_format(cls, doc: "HaystackDocument")
Document.from_langchain_format(cls, doc: "LCDocument")
Document.from_semantic_kernel_format(cls, doc: "MemoryRecord")
Document.get_doc_id(self)
Document.get_type(cls)
Document.to_embedchain_format(self)
Document.to_haystack_format(self)
Document.to_langchain_format(self)
Document.to_semantic_kernel_format(self)
Document.to_vectorflow(self, client: Any)
ImageDocument.class_name(cls)
ImageNode.class_name(cls)
ImageNode.get_type(cls)
ImageNode.resolve_image(self)
IndexNode.class_name(cls)
IndexNode.dict(self, **kwargs: Any)
IndexNode.from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore).from_dict(data, **kwargs)"obj", None)parsed_obj  =  Noneobj, str))
IndexNode.from_text_node(cls, node: TextNode, index_id: str, )
IndexNode.get_type(cls)
NodeWithScore.__str__(self)
NodeWithScore.class_name(cls)
NodeWithScore.embedding(self)
NodeWithScore.get_content(self, metadata_mode: MetadataMode  =  MetadataMode.NONE)
NodeWithScore.get_embedding(self)
NodeWithScore.get_score(self, raise_error: bool  =  False)
NodeWithScore.get_text(self)
NodeWithScore.id_(self)
NodeWithScore.metadata(self)
NodeWithScore.node_id(self)
NodeWithScore.text(self)
QueryBundle.__str__(self)
QueryBundle.embedding_image(self)
QueryBundle.embedding_strs(self)
RelatedNodeInfo.class_name(cls)
TextNode.class_name(cls)
TextNode.get_content(self, metadata_mode: MetadataMode  =  MetadataMode.NONE)
TextNode.get_metadata_str(self, mode: MetadataMode  =  MetadataMode.ALL)
TextNode.get_node_info(self)
TextNode.get_text(self)
TextNode.get_type(cls)
TextNode.hash(self)
TextNode.node_info(self)
TextNode.set_content(self, value: str)
TransformComponent.__call__(self, nodes: List["BaseNode"], **kwargs: Any)


repos/llama_index/llama-index-core/llama_index/core/service_context.py
-------------------------functions----------------------
_get_default_node_parser(chunk_size: int  =  DEFAULT_CHUNK_SIZE, chunk_overlap: int  =  SENTENCE_CHUNK_OVERLAP, callback_manager: Optional[CallbackManager]  =  None, )
_get_default_prompt_helper(llm_metadata: LLMMetadata, context_window: Optional[int]  =  None, num_output: Optional[int]  =  None, )
set_global_service_context(service_context: Optional[ServiceContext])

-------------------------methods----------------------
ServiceContext.from_defaults(cls, llm_predictor: Optional[BaseLLMPredictor]  =  None, llm: Optional[LLMType]  =  "default", prompt_helper: Optional[PromptHelper]  =  None, embed_model: Optional[Any]  =  "default", node_parser: Optional[NodeParser]  =  None, text_splitter: Optional[TextSplitter]  =  None, transformations: Optional[List[TransformComponent]]  =  None, llama_logger: Optional[LlamaLogger]  =  None, callback_manager: Optional[CallbackManager]  =  None, system_prompt: Optional[str]  =  None, query_wrapper_prompt: Optional[BasePromptTemplate]  =  None, pydantic_program_mode: PydanticProgramMode  =  PydanticProgramMode.DEFAULT, chunk_size: Optional[int]  =  None, chunk_overlap: Optional[int]  =  None, context_window: Optional[int]  =  None, num_output: Optional[int]  =  None, chunk_size_limit: Optional[int]  =  None, )
ServiceContext.from_dict(cls, data: dict)
ServiceContext.from_service_context(cls, service_context: "ServiceContext", llm_predictor: Optional[BaseLLMPredictor]  =  None, llm: Optional[LLMType]  =  "default", prompt_helper: Optional[PromptHelper]  =  None, embed_model: Optional[Any]  =  "default", node_parser: Optional[NodeParser]  =  None, text_splitter: Optional[TextSplitter]  =  None, transformations: Optional[List[TransformComponent]]  =  None, llama_logger: Optional[LlamaLogger]  =  None, callback_manager: Optional[CallbackManager]  =  None, system_prompt: Optional[str]  =  None, query_wrapper_prompt: Optional[BasePromptTemplate]  =  None, chunk_size: Optional[int]  =  None, chunk_overlap: Optional[int]  =  None, context_window: Optional[int]  =  None, num_output: Optional[int]  =  None, chunk_size_limit: Optional[int]  =  None, )
ServiceContext.llm(self)
ServiceContext.node_parser(self)
ServiceContext.to_dict(self)


repos/llama_index/llama-index-core/llama_index/core/settings.py
-------------------------functions----------------------
callback_manager_from_settings_or_context(settings: _Settings, context: Optional["ServiceContext"])
embed_model_from_settings_or_context(settings: _Settings, context: Optional["ServiceContext"])
llm_from_settings_or_context(settings: _Settings, context: Optional["ServiceContext"])
node_parser_from_settings_or_context(settings: _Settings, context: Optional["ServiceContext"])
transformations_from_settings_or_context(settings: _Settings, context: Optional["ServiceContext"])

-------------------------methods----------------------
_Settings.callback_manager(self)
_Settings.callback_manager(self)
_Settings.chunk_overlap(self)
_Settings.chunk_overlap(self)
_Settings.chunk_size(self)
_Settings.chunk_size(self)
_Settings.context_window(self)
_Settings.context_window(self)
_Settings.embed_model(self)
_Settings.embed_model(self)
_Settings.global_handler(self)
_Settings.global_handler(self)
_Settings.llm(self)
_Settings.llm(self)
_Settings.node_parser(self)
_Settings.node_parser(self)
_Settings.num_output(self)
_Settings.num_output(self)
_Settings.prompt_helper(self)
_Settings.prompt_helper(self)
_Settings.pydantic_program_mode(self)
_Settings.pydantic_program_mode(self)
_Settings.text_splitter(self)
_Settings.text_splitter(self)
_Settings.tokenizer(self)
_Settings.tokenizer(self)
_Settings.transformations(self)
_Settings.transformations(self)


repos/llama_index/llama-index-core/llama_index/core/types.py
-------------------------methods----------------------
BaseOutputParser.__modify_schema__(cls, schema: Dict[str, Any])
BaseOutputParser.format(self, query: str)
BaseOutputParser.format_messages(self, messages: List[ChatMessage])
BaseOutputParser.parse(self, output: str)
BasePydanticProgram.__call__(self, *args: Any, **kwds: Any)
BasePydanticProgram.output_cls(self)


repos/llama_index/llama-index-core/llama_index/core/utils.py
-------------------------functions----------------------
_get_colored_text(text: str, color: str)
add_sync_version(func: Any)
concat_dirs(dirname: str, basename: str)
count_tokens(text: str)
get_cache_dir()
get_color_mapping(items: List[str], use_llama_index_colors: bool  =  True)
get_new_id(d: Set)
get_new_int_id(d: Set)
get_tokenizer()
get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str)
get_transformer_tokenizer_fn(model_name: str)
infer_torch_device()
iter_batch(iterable: Union[Iterable, Generator], size: int)
print_text(text: str, color: Optional[str]  =  None, end: str  =  "")
retry_on_exceptions_with_backoff(lambda_fn: Callable, errors_to_retry: List[ErrorToRetry], max_tries: int  =  10, min_backoff_secs: float  =  0.5, max_backoff_secs: float  =  60.0, )
set_global_tokenizer(tokenizer: Union[Tokenizer, Callable[[str], list]])
temp_set_attrs(obj: Any, **kwargs: Any)
truncate_text(text: str, max_length: int)
unit_generator(x: Any)

-------------------------methods----------------------
GlobalsHelper.__init__(self)
GlobalsHelper.stopwords(self)
Tokenizer.encode(self, text: str, *args: Any, **kwargs: Any)


repos/llama_index/llama-index-core/tests/__init__.py


repos/llama_index/llama-index-core/tests/agent/__init__.py


repos/llama_index/llama-index-core/tests/callbacks/__init__.py


repos/llama_index/llama-index-core/tests/callbacks/test_llama_debug.py
-------------------------functions----------------------
test_flush_events()
test_get_event_stats()
test_ignore_events()
test_on_event_end()
test_on_event_start()



repos/llama_index/llama-index-core/tests/callbacks/test_token_counter.py
-------------------------functions----------------------
test_on_event_end()
test_on_event_start()



repos/llama_index/llama-index-core/tests/chat_engine/__init__.py


repos/llama_index/llama-index-core/tests/chat_engine/test_condense_plus_context.py
-------------------------functions----------------------
override_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any)
test_condense_plus_context_chat_engine(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-core/tests/chat_engine/test_condense_question.py
-------------------------functions----------------------
test_condense_question_chat_engine(mock_service_context: ServiceContext, )
test_condense_question_chat_engine_with_init_history(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-core/tests/chat_engine/test_simple.py
-------------------------functions----------------------
test_simple_chat_engine(mock_service_context: ServiceContext, )
test_simple_chat_engine_with_init_history(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-core/tests/conftest.py
-------------------------functions----------------------
allow_networking(monkeypatch: pytest.MonkeyPatch)
mock_llm()
mock_openai_credentials()
mock_service_context(patch_token_text_splitter: Any, patch_llm_predictor: Any, )
patch_llm_predictor(monkeypatch: pytest.MonkeyPatch)
patch_token_text_splitter(monkeypatch: pytest.MonkeyPatch)
pytest_addoption(parser: pytest.Parser)
pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item])
pytest_configure(config: pytest.Config)
set_env_vars()

-------------------------methods----------------------
CachedOpenAIApiKeys.__enter__(self)
CachedOpenAIApiKeys.__exit__(self, *exc: object)
CachedOpenAIApiKeys.__init__(self, set_env_key_to: Optional[str]  =  "", set_library_key_to: Optional[str]  =  None, set_fake_key: bool  =  False, set_env_type_to: Optional[str]  =  "", set_library_type_to: str  =  "open_ai", # default value in openai package)


repos/llama_index/llama-index-core/tests/embeddings/__init__.py


repos/llama_index/llama-index-core/tests/embeddings/test_base.py
-------------------------functions----------------------
mock_get_text_embedding(text: str)
mock_get_text_embeddings(texts: List[str])
test_embedding_similarity()
test_embedding_similarity_euclidean()
test_get_text_embeddings(_mock_get_text_embeddings: Any, _mock_get_text_embedding: Any)
test_mean_agg()



repos/llama_index/llama-index-core/tests/embeddings/test_utils.py
-------------------------functions----------------------
test_resolve_embed_model(monkeypatch: MonkeyPatch)



repos/llama_index/llama-index-core/tests/embeddings/todo_hf_test_utils.py
-------------------------functions----------------------
mock_hf_embeddings(self: Any, *args: Any, **kwargs: Dict[str, Any])
mock_openai_embeddings(self: Any, *args: Any, **kwargs: Dict[str, Any])
test_resolve_embed_model(monkeypatch: MonkeyPatch)



repos/llama_index/llama-index-core/tests/evaluation/test_base.py
-------------------------functions----------------------
test_evaluator_basic()

-------------------------methods----------------------
MockEvaluator.__init__(self, mock_score: float  =  1.0, mock_passing: bool  =  True, mock_feedback: str  =  "test feedback", )
MockEvaluator._get_prompts(self)
MockEvaluator._update_prompts(self, prompts: PromptDictType)


repos/llama_index/llama-index-core/tests/evaluation/test_batch_runner.py
-------------------------functions----------------------
get_eval_results(key, eval_results)
test_batch_runner()

-------------------------methods----------------------
MockEvaluator.__init__(self, mock_score: float  =  1.0, mock_passing: bool  =  True, mock_feedback: str  =  "test feedback", )
MockEvaluator._get_prompts(self)
MockEvaluator._update_prompts(self, prompts: PromptDictType)


repos/llama_index/llama-index-core/tests/evaluation/test_dataset_generation.py
-------------------------functions----------------------
test_dataset_generation(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-core/tests/evaluation/test_platform_eval.py
-------------------------functions----------------------
test_upload_eval_dataset()



repos/llama_index/llama-index-core/tests/evaluation/test_rr_mrr_hitrate.py
-------------------------functions----------------------
test_exceptions(expected_ids, retrieved_ids, use_granular)
test_hit_rate(expected_ids, retrieved_ids, use_granular, expected_result)
test_mrr(expected_ids, retrieved_ids, use_granular, expected_result)



repos/llama_index/llama-index-core/tests/indices/__init__.py


repos/llama_index/llama-index-core/tests/indices/conftest.py
-------------------------functions----------------------
documents()
nodes()



repos/llama_index/llama-index-core/tests/indices/test_loading.py
-------------------------functions----------------------
test_load_index_from_storage_multiple(nodes: List[BaseNode], tmp_path: Path, mock_service_context: ServiceContext, )
test_load_index_from_storage_retrieval_result_identical(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext, )
test_load_index_from_storage_simple(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext)
test_load_index_query_engine_service_context(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext, )



repos/llama_index/llama-index-core/tests/indices/test_loading_graph.py
-------------------------functions----------------------
test_load_graph_from_storage_simple(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext, )



repos/llama_index/llama-index-core/tests/indices/test_prompt_helper.py
-------------------------functions----------------------
test_get_biggest_prompt()
test_get_chunk_size(prompt: str, chunk_size_limit: Optional[int], num_chunks: int, padding: int, expected: Union[int, Type[Exception]], )
test_get_numbered_text_from_nodes()
test_get_text_splitter()
test_get_text_splitter_partial()
test_repack()
test_truncate()



repos/llama_index/llama-index-core/tests/indices/test_service_context.py
-------------------------functions----------------------
test_service_context_serialize()



repos/llama_index/llama-index-core/tests/indices/test_utils.py
-------------------------functions----------------------
test_expand_tokens_with_subtokens()



repos/llama_index/llama-index-core/tests/ingestion/test_cache.py
-------------------------functions----------------------
test_cache()
test_cache_clear()

-------------------------methods----------------------
DummyTransform.__call__(self, nodes: List[BaseNode], **kwargs: Any)


repos/llama_index/llama-index-core/tests/ingestion/test_data_sinks.py
-------------------------functions----------------------
test_build_configured_data_sink()
test_can_build_configured_data_sink_from_component()
test_can_generate_schema_for_data_sink_component_type(configurable_data_sink_type: ConfigurableDataSinks, )
test_unique_configurable_data_sink_names()



repos/llama_index/llama-index-core/tests/ingestion/test_data_sources.py
-------------------------functions----------------------
test_build_configured_data_source()
test_can_build_configured_data_source_from_component()
test_can_generate_schema_for_data_source_component_type(configurable_data_source_type: ConfigurableDataSources, )
test_unique_configurable_data_source_names()



repos/llama_index/llama-index-core/tests/ingestion/test_pipeline.py
-------------------------functions----------------------
teardown_function()
test_build_pipeline()
test_pipeline_dedup_duplicates_only()
test_pipeline_parallel()
test_pipeline_update()
test_run_pipeline()
test_save_load_pipeline()
test_save_load_pipeline_without_docstore()



repos/llama_index/llama-index-core/tests/ingestion/test_transformations.py
-------------------------functions----------------------
test_build_configured_transformation()
test_can_build_configured_transform_from_component()
test_can_generate_schema_for_transformation_component_type(configurable_transformation_type: ConfigurableTransformations, )
test_unique_configurable_transformations_names()



repos/llama_index/llama-index-core/tests/instrumentation/test_dispatcher.py
-------------------------functions----------------------
func(a, b = 3, **kwargs)
func_exc(a, b = 3, c = 4, **kwargs)
func_with_event(a, b = 3, **kwargs)
test_dispatcher_fire_event(mock_uuid: MagicMock, mock_span_enter: MagicMock, mock_span_drop: MagicMock, mock_span_exit: MagicMock, )
test_dispatcher_fire_event_with_instance(mock_uuid, mock_span_enter, mock_span_drop, mock_span_exit)
test_dispatcher_span_args(mock_uuid, mock_span_enter, mock_span_exit)
test_dispatcher_span_args_with_instance(mock_uuid, mock_span_enter, mock_span_exit)
test_dispatcher_span_drop_args(mock_uuid: MagicMock, mock_span_enter: MagicMock, mock_span_drop: MagicMock, mock_span_exit: MagicMock, )
test_dispatcher_span_drop_args(mock_uuid: MagicMock, mock_span_enter: MagicMock, mock_span_drop: MagicMock, mock_span_exit: MagicMock, )

-------------------------methods----------------------
_TestEndEvent.class_name(cls)
_TestEventHandler.class_name(cls)
_TestEventHandler.handle(self, e: BaseEvent)
_TestObject.func(self, a, b = 3, **kwargs)
_TestObject.func_exc(self, a, b = 3, c = 4, **kwargs)
_TestObject.func_with_event(self, a, b = 3, **kwargs)
_TestStartEvent.class_name(cls)


repos/llama_index/llama-index-core/tests/instrumentation/test_manager.py
-------------------------functions----------------------
test_root_manager_add_dispatcher()



repos/llama_index/llama-index-core/tests/llms/__init__.py


repos/llama_index/llama-index-core/tests/llms/test_callbacks.py
-------------------------functions----------------------
llm()
nonyielding_llm()
prompt()
test_llm_complete_prompt_arg(llm: LLM, prompt: str)
test_llm_complete_prompt_kwarg(llm: LLM, prompt: str)
test_llm_complete_throws_if_duplicate_prompt(llm: LLM, prompt: str)
test_llm_complete_throws_if_no_prompt(llm: LLM)
test_llm_stream_chat_handles_nonyielding_stream(nonyielding_llm: LLM, prompt: str)
test_llm_stream_complete_prompt_arg(llm: LLM, prompt: str)
test_llm_stream_complete_prompt_kwarg(llm: LLM, prompt: str)
test_llm_stream_complete_throws_if_duplicate_prompt(llm: LLM, prompt: str)
test_llm_stream_complete_throws_if_no_prompt(llm: LLM)



repos/llama_index/llama-index-core/tests/llms/test_custom.py
-------------------------functions----------------------
test_basic()
test_streaming()

-------------------------methods----------------------
TestLLM.__init__(self)
TestLLM.complete(self, prompt: str, formatted: bool  =  False, **kwargs: Any)
TestLLM.metadata(self)
TestLLM.stream_complete(self, prompt: str, formatted: bool  =  False, **kwargs: Any)


repos/llama_index/llama-index-core/tests/memory/test_chat_memory_buffer.py
-------------------------functions----------------------
test_dict_save_load()
test_get_when_initial_tokens_exceed_limit_raises_value_error()
test_get_when_initial_tokens_less_than_limit_returns_history()
test_get_when_initial_tokens_same_as_limit_removes_message()
test_get_when_space_for_all_but_first_message_removes_first_message_and_answer() -> (None)
test_get_when_space_for_assistant_message_removes_assistant_message_at_start_of_history() -> (None)
test_get_when_space_for_second_message_and_answer_removes_only_first_message_and_answer() -> (None)
test_max_tokens()
test_pickle()
test_put_get()
test_set()
test_string_save_load()



repos/llama_index/llama-index-core/tests/memory/test_chat_summary_memory_buffer.py
-------------------------functions----------------------
_get_role_alternating_order(i: int)
summarizer_llm()
test_assistant_never_first_message(summarizer_llm)
test_assistant_tool_pairs(summarizer_llm)
test_dict_save_load(summarizer_llm)
test_get_when_initial_tokens_exceed_limit_raises_value_error()
test_get_when_initial_tokens_less_than_limit_returns_history()
test_max_tokens_with_summarizer(summarizer_llm)
test_max_tokens_without_summarizer()
test_pickle()
test_put_get(summarizer_llm)
test_put_get_summarize_long_message(summarizer_llm)
test_put_get_summarize_part_of_conversation(summarizer_llm)
test_set()
test_string_save_load(summarizer_llm)

-------------------------methods----------------------
MockSummarizerLLM.__init__(self, responses: List[ChatMessage], max_tokens: int  =  512)
MockSummarizerLLM.chat(self, messages: Sequence[ChatMessage], **kwargs: Any)
MockSummarizerLLM.get_role_count(self, role: MessageRole)
MockSummarizerLLM.set_max_tokens(self, max_tokens)


repos/llama_index/llama-index-core/tests/mock_utils/__init__.py


repos/llama_index/llama-index-core/tests/mock_utils/mock_predict.py
-------------------------functions----------------------
_mock_answer(prompt_args: Dict)
_mock_choice_select(prompt_args: Dict)
_mock_conversation(prompt_args: Dict)
_mock_decompose_query(prompt_args: Dict)
_mock_input(prompt_args: Dict)
_mock_insert_predict()
_mock_keyword_extract(prompt_args: Dict)
_mock_kg_triplet_extract(prompt_args: Dict)
_mock_multi_select(prompt_args: Dict)
_mock_pandas(prompt_args: Dict)
_mock_query_keyword_extract(prompt_args: Dict)
_mock_query_select()
_mock_refine(prompt_args: Dict)
_mock_schema_extract(prompt_args: Dict)
_mock_single_select()
_mock_sql_response_synthesis(prompt_args: Dict)
_mock_sql_response_synthesis_v2(prompt_args: Dict)
_mock_sub_questions()
_mock_summary_predict(prompt_args: Dict)
_mock_text_to_sql(prompt_args: Dict)
mock_llmpredictor_predict(prompt: BasePromptTemplate, **prompt_args: Any)
patch_llmpredictor_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any)



repos/llama_index/llama-index-core/tests/mock_utils/mock_prompts.py


repos/llama_index/llama-index-core/tests/mock_utils/mock_text_splitter.py
-------------------------functions----------------------
mock_token_splitter_newline(text: str, metadata_str: Optional[str]  =  None)
patch_token_splitter_newline(self: Any, text: str, metadata_str: Optional[str]  =  None)



repos/llama_index/llama-index-core/tests/mock_utils/mock_utils.py
-------------------------functions----------------------
mock_extract_keywords(text_chunk: str, max_keywords: Optional[int]  =  None, filter_stopwords: bool  =  True)
mock_extract_keywords_response(text_chunk: str, max_keywords: Optional[int]  =  None, filter_stopwords: bool  =  True)
mock_extract_kg_triplets_response(text_chunk: str, max_triplets: Optional[int]  =  None)
mock_tokenizer(text: str)



repos/llama_index/llama-index-core/tests/node_parser/metadata_extractor.py
-------------------------functions----------------------
test_metadata_extractor(mock_service_context: ServiceContext)



repos/llama_index/llama-index-core/tests/node_parser/sentence_window.py
-------------------------functions----------------------
test_split_and_window()



repos/llama_index/llama-index-core/tests/node_parser/test_file.py
-------------------------functions----------------------
test_unsupported_extension()



repos/llama_index/llama-index-core/tests/node_parser/test_hierarchical.py
-------------------------functions----------------------
nodes()
test_get_child_nodes(nodes: list)
test_get_deeper_nodes(nodes: list)
test_get_deeper_nodes_with_negative_depth(nodes: list)
test_get_deeper_nodes_with_no_root_nodes(nodes: list)
test_get_leaf_nodes(nodes: list)
test_get_root_nodes(nodes: list)
test_get_root_nodes_empty(nodes: list)



repos/llama_index/llama-index-core/tests/node_parser/test_html.py
-------------------------functions----------------------
test_multiple_tags_splits()
test_neighbor_tags_splits()
test_nesting_tags_splits()
test_no_splits()
test_single_splits()



repos/llama_index/llama-index-core/tests/node_parser/test_json.py
-------------------------functions----------------------
test_split_empty_text()
test_split_invalid_json()
test_split_valid_dict_json()
test_split_valid_json()
test_split_valid_json_defaults()



repos/llama_index/llama-index-core/tests/node_parser/test_markdown.py
-------------------------functions----------------------
test_header_metadata()
test_header_splits()
test_header_splits_with_indented_code_blocks()
test_non_header_splits()
test_pre_header_content()



repos/llama_index/llama-index-core/tests/node_parser/test_markdown_element.py
-------------------------functions----------------------
test_complex_md()
test_extract_ref_doc_id()
test_llama2_bad_md()
test_md_table_extraction()
test_md_table_extraction_broken_table()
test_start_end_char_idx()



repos/llama_index/llama-index-core/tests/node_parser/test_semantic_splitter.py
-------------------------functions----------------------
test_grouped_semantically()
test_split_and_permutated()

-------------------------methods----------------------
MockEmbedding._get_query_embedding(self, query: str)
MockEmbedding._get_text_embedding(self, text: str)
MockEmbedding.class_name(cls)


repos/llama_index/llama-index-core/tests/node_parser/test_unstructured.py
-------------------------functions----------------------
test_html_table_extraction()



repos/llama_index/llama-index-core/tests/objects/__init__.py


repos/llama_index/llama-index-core/tests/objects/test_base.py
-------------------------functions----------------------
test_object_index(mock_service_context: ServiceContext)
test_object_index_default_mapping(mock_service_context: ServiceContext)
test_object_index_fn_mapping(mock_service_context: ServiceContext)
test_object_index_persist(mock_service_context: ServiceContext)
test_object_index_with_tools(mock_service_context: ServiceContext)



repos/llama_index/llama-index-core/tests/objects/test_node_mapping.py
-------------------------functions----------------------
test_simple_object_node_mapping()
test_simple_object_node_mapping_persist()
test_sql_table_node_mapping_to_node(mocker: MockerFixture)
test_tool_object_node_mapping()

-------------------------methods----------------------
TestObject.__hash__(self)
TestObject.__str__(self)
TestSQLDatabase.__init__(self)


repos/llama_index/llama-index-core/tests/output_parsers/__init__.py


repos/llama_index/llama-index-core/tests/output_parsers/test_base.py
-------------------------functions----------------------
test_lc_output_parser()



repos/llama_index/llama-index-core/tests/output_parsers/test_pydantic.py
-------------------------functions----------------------
test_pydantic()
test_pydantic_format()



repos/llama_index/llama-index-core/tests/output_parsers/test_selection.py
-------------------------functions----------------------
output_parser()
test_format(output_parser: SelectionOutputParser)



repos/llama_index/llama-index-core/tests/output_parsers/test_utils.py
-------------------------functions----------------------
test_extract_json_str()



repos/llama_index/llama-index-core/tests/playground/__init__.py


repos/llama_index/llama-index-core/tests/playground/test_base.py
-------------------------functions----------------------
test_from_docs(mock_service_context: ServiceContext, )
test_get_set_compare(mock_service_context: ServiceContext, )
test_validation()

-------------------------methods----------------------
MockEmbedding._get_query_embedding(self, query: str)
MockEmbedding._get_text_embedding(self, text: str)
MockEmbedding.class_name(cls)


repos/llama_index/llama-index-core/tests/postprocessor/__init__.py


repos/llama_index/llama-index-core/tests/postprocessor/test_base.py
-------------------------functions----------------------
test_embedding_recency_postprocessor(mock_service_context: ServiceContext, )
test_fixed_recency_postprocessor(mock_service_context: ServiceContext, )
test_forward_back_processor(tmp_path: Path)
test_keyword_postprocessor()
test_keyword_postprocessor_for_non_english()
test_time_weighted_postprocessor()



repos/llama_index/llama-index-core/tests/postprocessor/test_llm_rerank.py
-------------------------functions----------------------
mock_format_node_batch_fn(nodes: List[BaseNode])
mock_llmpredictor_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any)
test_llm_rerank(mock_service_context: ServiceContext)



repos/llama_index/llama-index-core/tests/postprocessor/test_metadata_replacement.py
-------------------------functions----------------------
test_metadata_replacement()



repos/llama_index/llama-index-core/tests/postprocessor/test_optimizer.py
-------------------------functions----------------------
mock_get_text_embedding(text: str)
mock_get_text_embedding_chinese(text: str)
mock_get_text_embeddings(texts: List[str])
mock_get_text_embeddings_chinese(texts: List[str])
mock_tokenizer_fn(text: str)
mock_tokenizer_fn2(text: str)
test_optimizer(_mock_embeds: Any, _mock_embed: Any)



repos/llama_index/llama-index-core/tests/postprocessor/test_rankgpt_rerank.py
-------------------------functions----------------------
mock_rankgpt_chat(self: Any, messages, **kwargs: Any)
test_rankgpt_rerank()



repos/llama_index/llama-index-core/tests/program/__init__.py


repos/llama_index/llama-index-core/tests/program/test_function_program.py
-------------------------functions----------------------
_get_mock_album_response(allow_parallel_tool_calls: bool  =  False, )
test_function_program()
test_function_program_multiple()

-------------------------methods----------------------
MockLLM.metadata(self)
MockLLM.predict_and_call(self, tools: List["BaseTool"], user_msg: Optional[Union[str, ChatMessage]]  =  None, chat_history: Optional[List[ChatMessage]]  =  None, verbose: bool  =  False, allow_parallel_tool_calls: bool  =  False, **kwargs: Any, )


repos/llama_index/llama-index-core/tests/program/test_llm_program.py
-------------------------functions----------------------
test_llm_program()
test_llm_program_with_messages()
test_llm_program_with_messages_and_chat()

-------------------------methods----------------------
MockChatLLM.chat(self, prompt: str)
MockChatLLM.metadata(self)
MockLLM.complete(self, prompt: str)
MockLLM.metadata(self)


repos/llama_index/llama-index-core/tests/program/test_multi_modal_llm_program.py
-------------------------functions----------------------
test_multi_modal_llm_program()

-------------------------methods----------------------
MockMultiModalLLM.complete(self, prompt: str, image_documents: Sequence[ImageDocument])
MockMultiModalLLM.metadata(self)


repos/llama_index/llama-index-core/tests/prompts/__init__.py


repos/llama_index/llama-index-core/tests/prompts/test_base.py
-------------------------functions----------------------
output_parser()
test_chat_template()
test_chat_template_output_parser(output_parser: BaseOutputParser)
test_function_mappings()
test_selector_template()
test_template()
test_template_output_parser(output_parser: BaseOutputParser)
test_template_var_mappings()

-------------------------methods----------------------
MockOutputParser.__init__(self, format_string: str)
MockOutputParser.format(self, query: str)
MockOutputParser.parse(self, output: str)


repos/llama_index/llama-index-core/tests/prompts/test_guidance_utils.py
-------------------------functions----------------------
test_convert_pydantic_to_guidance_output_template_nested()
test_convert_pydantic_to_guidance_output_template_simple()
test_convert_to_handlebars()



repos/llama_index/llama-index-core/tests/prompts/test_mixin.py
-------------------------functions----------------------
test_prompt_mixin()

-------------------------methods----------------------
MockObject1.__init__(self)
MockObject1._get_prompt_modules(self)
MockObject1._get_prompts(self)
MockObject1._update_prompts(self, prompts: PromptDictType)
MockObject2.__init__(self)
MockObject2._get_prompt_modules(self)
MockObject2._get_prompts(self)
MockObject2._update_prompts(self, prompts: PromptDictType)


repos/llama_index/llama-index-core/tests/prompts/test_utils.py
-------------------------functions----------------------
test_get_template_vars()



repos/llama_index/llama-index-core/tests/query_engine/test_cogniswitch_query_engine.py
-------------------------functions----------------------
query_engine()
test_query_knowledge_successful(mock_post: Any, query_engine: CogniswitchQueryEngine)
test_query_knowledge_unsuccessful(mock_post: Any, query_engine: CogniswitchQueryEngine)



repos/llama_index/llama-index-core/tests/query_engine/test_retriever_query_engine.py
-------------------------functions----------------------
test_query_engine_falls_back_to_inheriting_retrievers_service_context()



repos/llama_index/llama-index-core/tests/query_pipeline/__init__.py


repos/llama_index/llama-index-core/tests/query_pipeline/test_components.py
-------------------------functions----------------------
bar_fn(a: Any, b: Any)
foo_fn(a: int, b: int  =  1, c: int  =  2)
sum_fn(a: List[int])
test_arg_component()
test_fn_components()
test_fn_pipeline()
test_kwarg_component()
test_selector_component()

-------------------------methods----------------------
MockSelector._get_prompts(self)
MockSelector._select(self, choices: Sequence[ToolMetadata], query: QueryBundle)
MockSelector._update_prompts()


repos/llama_index/llama-index-core/tests/query_pipeline/test_query.py
-------------------------functions----------------------
test_query_pipeline_batch_chain_str()
test_query_pipeline_chain()
test_query_pipeline_chain_str()
test_query_pipeline_chain_str_intermediate_output()
test_query_pipeline_conditional_edges()
test_query_pipeline_init()
test_query_pipeline_input_component()
test_query_pipeline_multi()
test_query_pipeline_multi_batch()
test_query_pipeline_multi_intermediate_output()
test_query_pipeline_partial()
test_query_pipeline_single_arg_inp()
test_query_pipeline_sub()
test_query_pipeline_super_conditional()

-------------------------methods----------------------
Chainable2._as_query_component(self, **kwargs: Any)
QueryComponent1._run_component(self, **kwargs: Any)
QueryComponent1._validate_component_inputs(self, input: Dict[str, Any])
QueryComponent1.input_keys(self)
QueryComponent1.output_keys(self)
QueryComponent1.set_callback_manager(self, callback_manager: Any)
QueryComponent2._run_component(self, **kwargs: Any)
QueryComponent2._validate_component_inputs(self, input: Dict[str, Any])
QueryComponent2.input_keys(self)
QueryComponent2.output_keys(self)
QueryComponent2.set_callback_manager(self, callback_manager: Any)
QueryComponent3._run_component(self, **kwargs: Any)
QueryComponent3._validate_component_inputs(self, input: Dict[str, Any])
QueryComponent3.input_keys(self)
QueryComponent3.output_keys(self)
QueryComponent3.set_callback_manager(self, callback_manager: Any)


repos/llama_index/llama-index-core/tests/question_gen/__init__.py


repos/llama_index/llama-index-core/tests/question_gen/test_llm_generators.py
-------------------------functions----------------------
test_llm_question_gen(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-core/tests/readers/__init__.py


repos/llama_index/llama-index-core/tests/readers/test_json.py
-------------------------functions----------------------
test_basic()
test_clean_json()
test_collapse_length()
test_jsonl()
test_levels_back0()



repos/llama_index/llama-index-core/tests/readers/test_load_reader.py
-------------------------functions----------------------
test_loading_readers()



repos/llama_index/llama-index-core/tests/readers/test_string_iterable.py
-------------------------functions----------------------
test_load()



repos/llama_index/llama-index-core/tests/response_synthesizers/__init__.py


repos/llama_index/llama-index-core/tests/response_synthesizers/test_refine.py
-------------------------functions----------------------
mock_refine_service_context(patch_llm_predictor: Any)
refine_instance(mock_refine_service_context: ServiceContext)
test_constructor_args(mock_refine_service_context: ServiceContext)

-------------------------methods----------------------
MockRefineProgram.__call__(self, *args: Any, context_str: Optional[str]  =  None, context_msg: Optional[str]  =  None, **kwargs: Any)
MockRefineProgram.__init__(self, input_to_query_satisfied: Dict[str, bool])
MockRefineProgram.output_cls(self)


repos/llama_index/llama-index-core/tests/retrievers/__init__.py


repos/llama_index/llama-index-core/tests/retrievers/test_composable_retriever.py
-------------------------functions----------------------
test_composable_retrieval()



repos/llama_index/llama-index-core/tests/selectors/test_llm_selectors.py
-------------------------functions----------------------
test_llm_multi_selector(mock_service_context: ServiceContext, )
test_llm_multi_selector_max_choices(mock_service_context: ServiceContext, )
test_llm_single_selector()



repos/llama_index/llama-index-core/tests/test_schema.py
-------------------------functions----------------------
node_with_score(text_node: TextNode)
test_node_with_score_passthrough(node_with_score: NodeWithScore)
test_text_node_hash()
text_node()



repos/llama_index/llama-index-core/tests/test_utils.py
-------------------------functions----------------------
fn_with_exception(exception_cls: Optional[Union[Type[Exception], Exception]])
test_get_color_mapping()
test_get_colored_text()
test_iter_batch()
test_print_text(capsys: CaptureFixture)
test_retry_on_conditional_exceptions()
test_retry_on_exceptions_with_backoff()
test_tokenizer()

-------------------------methods----------------------
ConditionalException.__init__(self, should_retry: bool)


repos/llama_index/llama-index-core/tests/text_splitter/__init__.py


repos/llama_index/llama-index-core/tests/text_splitter/test_code_splitter.py
-------------------------functions----------------------
baz()
baz()
baz()
foo()
foo()
foo()
test__py_custom_parser_code_splitter()
test_cpp_code_splitter()
test_html_code_splitter()
test_python_code_splitter()
test_start_end_char_idx()
test_tsx_code_splitter()
test_typescript_code_splitter()



repos/llama_index/llama-index-core/tests/text_splitter/test_sentence_splitter.py
-------------------------functions----------------------
test_chinese_text(chinese_text: str)
test_contiguous_text(contiguous_text: str)
test_edge_case()
test_overlap()
test_paragraphs()
test_sentences()
test_split_texts_multiple()
test_split_texts_singleton()
test_split_texts_with_metadata(english_text: str)
test_split_with_metadata(english_text: str)
test_start_end_char_idx()



repos/llama_index/llama-index-core/tests/text_splitter/test_token_splitter.py
-------------------------functions----------------------
test_contiguous_text(contiguous_text: str)
test_split_chinese(chinese_text: str)
test_split_long_token()
test_split_token()
test_split_with_metadata(english_text: str)
test_start_end_char_idx()
test_truncate_token()



repos/llama_index/llama-index-core/tests/token_predictor/__init__.py


repos/llama_index/llama-index-core/tests/token_predictor/test_base.py
-------------------------functions----------------------
test_token_predictor(mock_split: Any)



repos/llama_index/llama-index-core/tests/tools/__init__.py


repos/llama_index/llama-index-core/tests/tools/conftest.py
-------------------------functions----------------------
documents()



repos/llama_index/llama-index-core/tests/tools/test_base.py
-------------------------functions----------------------
test_function_tool()
test_function_tool_to_langchain()
test_retreiver_tool()
test_tool_fn_schema()
tmp_function(x: int)



repos/llama_index/llama-index-core/tests/tools/test_eval_query_engine_tool.py
-------------------------methods----------------------
MockEvaluator._get_prompts(self)
MockEvaluator._update_prompts(self, prompts_dict: PromptDictType)
MockQueryEngine.custom_query(self, query_str: str)
TestEvalQueryEngineTool.setUp(self)
TestEvalQueryEngineTool.test_eval_query_engine_tool_with_eval_failing(self)
TestEvalQueryEngineTool.test_eval_query_engine_tool_with_eval_passing(self)


repos/llama_index/llama-index-core/tests/tools/test_ondemand_loader.py
-------------------------functions----------------------
test_ondemand_loader_tool(tool: OnDemandLoaderTool, )
test_ondemand_loader_tool_langchain(tool: OnDemandLoaderTool, )
tool(mock_service_context: ServiceContext)



repos/llama_index/llama-index-core/tests/tools/test_query_engine_tool.py
-------------------------functions----------------------
test_query_engine_tool()

-------------------------methods----------------------
MockQueryEngine.custom_query(self, query_str: str)


repos/llama_index/llama-index-core/tests/tools/test_retriever_tool.py
-------------------------functions----------------------
test_retriever_tool()

-------------------------methods----------------------
MockPostProcessor._postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]  =  None, )
MockPostProcessor.class_name(cls)
MockRetriever._retrieve(self, query_str: str)


repos/llama_index/llama-index-core/tests/tools/test_utils.py
-------------------------functions----------------------
test_create_schema_from_function()
test_create_schema_from_function_with_field()



repos/llama_index/llama-index-core/tests/utilities/test_sql_wrapper.py
-------------------------functions----------------------
sql_database(request: pytest.FixtureRequest)
test_get_single_table_info(sql_database: SQLDatabase)
test_get_table_columns(sql_database: SQLDatabase)
test_init(sql_database: SQLDatabase)
test_insert_and_run_sql(sql_database: SQLDatabase)
test_long_string_no_truncation(sql_database: SQLDatabase)
test_run_sql_truncation(sql_database: SQLDatabase)



repos/llama_index/llama-index-core/tests/vector_stores/__init__.py


repos/llama_index/llama-index-core/tests/vector_stores/test_simple.py
-------------------------functions----------------------
_node_embeddings_for_test()

-------------------------methods----------------------
SimpleVectorStoreTest.test_clear(self)
SimpleVectorStoreTest.test_delete_nodes(self)
SimpleVectorStoreTest.test_delete_removes_document_from_query_results(self)
SimpleVectorStoreTest.test_query_with_contradictive_filter_returns_no_matches(self)
SimpleVectorStoreTest.test_query_with_exact_filters_returns_single_match(self)
SimpleVectorStoreTest.test_query_with_filter_applies_node_id_filter(self)
SimpleVectorStoreTest.test_query_with_filter_applies_top_k(self)
SimpleVectorStoreTest.test_query_with_filter_on_unknown_field_returns_no_matches(self)
SimpleVectorStoreTest.test_query_with_filters_returns_multiple_matches(self)
SimpleVectorStoreTest.test_query_with_filters_with_filter_condition(self)
SimpleVectorStoreTest.test_query_without_filters_returns_all_rows_sorted_by_similarity(self)


repos/llama_index/llama-index-experimental/llama_index/experimental/__init__.py


repos/llama_index/llama-index-experimental/llama_index/experimental/exec_utils.py
-------------------------functions----------------------
_contains_protected_access(code: str)
_get_restricted_globals(__globals: Union[dict, None])
_restricted_import(name: str, globals: Union[Mapping[str, object], None]  =  None, locals: Union[Mapping[str, object], None]  =  None, ), level: int  =  0, )
_verify_source_safety(__source: Union[str, bytes, CodeType])
safe_eval(__source: Union[str, bytes, CodeType], __globals: Union[Dict[str, Any], None]  =  None, __locals: Union[Mapping[str, object], None]  =  None, )
safe_exec(__source: Union[str, bytes, CodeType], __globals: Union[Dict[str, Any], None]  =  None, __locals: Union[Mapping[str, object], None]  =  None, )

-------------------------methods----------------------
DunderVisitor.__init__(self)
DunderVisitor.visit_Attribute(self, node: ast.Attribute)
DunderVisitor.visit_Name(self, node: ast.Name)


repos/llama_index/llama-index-experimental/tests/__init__.py


repos/llama_index/llama-index-experimental/tests/param_tuner/__init__.py


repos/llama_index/llama-index-experimental/tests/param_tuner/test_base.py
-------------------------functions----------------------
_mock_obj_function(param_dict: Dict)
test_param_tuner()



repos/llama_index/llama-index-experimental/tests/param_tuner/test_param_tuner_classes.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-experimental/tests/test_exec_utils.py
-------------------------functions----------------------
test_contains_protected_access()



repos/llama_index/llama-index-experimental/tests/test_pandas.py
-------------------------functions----------------------
_mock_predict(*args: Any, **kwargs: Any)
test_pandas_query_engine(monkeypatch: pytest.MonkeyPatch)



repos/llama_index/llama-index-finetuning/llama_index/finetuning/__init__.py


repos/llama_index/llama-index-finetuning/llama_index/finetuning/types.py


repos/llama_index/llama-index-finetuning/tests/__init__.py


repos/llama_index/llama-index-finetuning/tests/callbacks/__init__.py


repos/llama_index/llama-index-finetuning/tests/callbacks/test_callback_classes.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-finetuning/tests/cross_encoders/__init__.py


repos/llama_index/llama-index-finetuning/tests/cross_encoders/test_cross_encoder_classes.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-finetuning/tests/embeddings/__init__.py


repos/llama_index/llama-index-finetuning/tests/embeddings/test_embeddings_classes.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-finetuning/tests/gradient/__init__.py


repos/llama_index/llama-index-finetuning/tests/gradient/test_gradient_classes.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-finetuning/tests/openai/__init__.py


repos/llama_index/llama-index-finetuning/tests/openai/test_openai_classes.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-finetuning/tests/rerankers/__init__.py


repos/llama_index/llama-index-finetuning/tests/rerankers/test_rerankers_classes.py
-------------------------functions----------------------
test_classes()



repos/llama_index/llama-index-finetuning/tests/test_base.py
-------------------------functions----------------------
test_torch_imports()



repos/llama_index/llama-index-integrations/retrievers/llama-index-retrievers-bedrock/__init__.py


repos/llama_index/llama-index-legacy/llama_index/legacy/__init__.py


repos/llama_index/llama-index-legacy/llama_index/legacy/async_utils.py
-------------------------functions----------------------
asyncio_module(show_progress: bool  =  False)
chunks(iterable: Iterable, size: int)
get_asyncio_module(show_progress: bool  =  False)
run_async_tasks(tasks: List[Coroutine], show_progress: bool  =  False, progress_bar_desc: str  =  "Running async tasks", )



repos/llama_index/llama-index-legacy/llama_index/legacy/constants.py


repos/llama_index/llama-index-legacy/llama_index/legacy/exec_utils.py
-------------------------functions----------------------
_contains_protected_access(code: str)
_get_restricted_globals(__globals: Union[dict, None])
_restricted_import(name: str, globals: Union[Mapping[str, object], None]  =  None, locals: Union[Mapping[str, object], None]  =  None, ), level: int  =  0, )
_verify_source_safety(__source: Union[str, bytes, CodeType])
safe_eval(__source: Union[str, bytes, CodeType], __globals: Union[Dict[str, Any], None]  =  None, __locals: Union[Mapping[str, object], None]  =  None, )
safe_exec(__source: Union[str, bytes, CodeType], __globals: Union[Dict[str, Any], None]  =  None, __locals: Union[Mapping[str, object], None]  =  None, )

-------------------------methods----------------------
DunderVisitor.__init__(self)
DunderVisitor.visit_Attribute(self, node: ast.Attribute)
DunderVisitor.visit_Name(self, node: ast.Name)


repos/llama_index/llama-index-legacy/llama_index/legacy/img_utils.py
-------------------------functions----------------------
b64_2_img(data: str)
img_2_b64(image: Image, format: str  =  "JPEG")



repos/llama_index/llama-index-legacy/llama_index/legacy/schema.py
-------------------------methods----------------------
BaseComponent.__getstate__(self)
BaseComponent.__setstate__(self, state: Dict[str, Any])
BaseComponent.class_name(cls)
BaseComponent.dict(self, **kwargs: Any)
BaseComponent.from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignorekwargs, dict))
BaseComponent.from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignoredata_str)data, **kwargs))
BaseComponent.json(self, **kwargs: Any)
BaseComponent.to_dict(self, **kwargs: Any)
BaseComponent.to_json(self, **kwargs: Any)
BaseNode.__str__(self)
BaseNode.as_related_node_info(self)
BaseNode.child_nodes(self)
BaseNode.extra_info(self)
BaseNode.get_content(self, metadata_mode: MetadataMode  =  MetadataMode.ALL)
BaseNode.get_embedding(self)
BaseNode.get_metadata_str(self, mode: MetadataMode  =  MetadataMode.ALL)
BaseNode.get_type(cls)
BaseNode.hash(self)
BaseNode.next_node(self)
BaseNode.node_id(self)
BaseNode.node_id(self)
BaseNode.parent_node(self)
BaseNode.prev_node(self)
BaseNode.ref_doc_id(self)
BaseNode.set_content(self, value: Any)
BaseNode.source_node(self)
Document.__setattr__(self, name: str, value: object)
Document.__str__(self)
Document.class_name(cls)
Document.doc_id(self)
Document.example(cls)
Document.from_embedchain_format(cls, doc: Dict[str, Any])
Document.from_haystack_format(cls, doc: "HaystackDocument")
Document.from_langchain_format(cls, doc: "LCDocument")
Document.from_semantic_kernel_format(cls, doc: "MemoryRecord")
Document.get_doc_id(self)
Document.get_type(cls)
Document.to_embedchain_format(self)
Document.to_haystack_format(self)
Document.to_langchain_format(self)
Document.to_semantic_kernel_format(self)
Document.to_vectorflow(self, client: Any)
ImageDocument.class_name(cls)
ImageNode.class_name(cls)
ImageNode.get_type(cls)
ImageNode.resolve_image(self)
IndexNode.class_name(cls)
IndexNode.from_text_node(cls, node: TextNode, index_id: str, )
IndexNode.get_type(cls)
NodeWithScore.__str__(self)
NodeWithScore.class_name(cls)
NodeWithScore.embedding(self)
NodeWithScore.get_content(self, metadata_mode: MetadataMode  =  MetadataMode.NONE)
NodeWithScore.get_embedding(self)
NodeWithScore.get_score(self, raise_error: bool  =  False)
NodeWithScore.get_text(self)
NodeWithScore.id_(self)
NodeWithScore.metadata(self)
NodeWithScore.node_id(self)
NodeWithScore.text(self)
QueryBundle.__str__(self)
QueryBundle.embedding_image(self)
QueryBundle.embedding_strs(self)
RelatedNodeInfo.class_name(cls)
TextNode.class_name(cls)
TextNode.get_content(self, metadata_mode: MetadataMode  =  MetadataMode.NONE)
TextNode.get_metadata_str(self, mode: MetadataMode  =  MetadataMode.ALL)
TextNode.get_node_info(self)
TextNode.get_text(self)
TextNode.get_type(cls)
TextNode.hash(self)
TextNode.node_info(self)
TextNode.set_content(self, value: str)
TransformComponent.__call__(self, nodes: List["BaseNode"], **kwargs: Any)


repos/llama_index/llama-index-legacy/llama_index/legacy/service_context.py
-------------------------functions----------------------
_get_default_node_parser(chunk_size: int  =  DEFAULT_CHUNK_SIZE, chunk_overlap: int  =  SENTENCE_CHUNK_OVERLAP, callback_manager: Optional[CallbackManager]  =  None, )
_get_default_prompt_helper(llm_metadata: LLMMetadata, context_window: Optional[int]  =  None, num_output: Optional[int]  =  None, )
set_global_service_context(service_context: Optional[ServiceContext])

-------------------------methods----------------------
ServiceContext.from_defaults(cls, llm_predictor: Optional[BaseLLMPredictor]  =  None, llm: Optional[LLMType]  =  "default", prompt_helper: Optional[PromptHelper]  =  None, embed_model: Optional[Any]  =  "default", node_parser: Optional[NodeParser]  =  None, text_splitter: Optional[TextSplitter]  =  None, transformations: Optional[List[TransformComponent]]  =  None, llama_logger: Optional[LlamaLogger]  =  None, callback_manager: Optional[CallbackManager]  =  None, system_prompt: Optional[str]  =  None, query_wrapper_prompt: Optional[BasePromptTemplate]  =  None, pydantic_program_mode: PydanticProgramMode  =  PydanticProgramMode.DEFAULT, chunk_size: Optional[int]  =  None, chunk_overlap: Optional[int]  =  None, context_window: Optional[int]  =  None, num_output: Optional[int]  =  None, chunk_size_limit: Optional[int]  =  None, )
ServiceContext.from_dict(cls, data: dict)
ServiceContext.from_service_context(cls, service_context: "ServiceContext", llm_predictor: Optional[BaseLLMPredictor]  =  None, llm: Optional[LLMType]  =  "default", prompt_helper: Optional[PromptHelper]  =  None, embed_model: Optional[Any]  =  "default", node_parser: Optional[NodeParser]  =  None, text_splitter: Optional[TextSplitter]  =  None, transformations: Optional[List[TransformComponent]]  =  None, llama_logger: Optional[LlamaLogger]  =  None, callback_manager: Optional[CallbackManager]  =  None, system_prompt: Optional[str]  =  None, query_wrapper_prompt: Optional[BasePromptTemplate]  =  None, chunk_size: Optional[int]  =  None, chunk_overlap: Optional[int]  =  None, context_window: Optional[int]  =  None, num_output: Optional[int]  =  None, chunk_size_limit: Optional[int]  =  None, )
ServiceContext.llm(self)
ServiceContext.node_parser(self)
ServiceContext.to_dict(self)


repos/llama_index/llama-index-legacy/llama_index/legacy/types.py
-------------------------methods----------------------
BaseOutputParser.format(self, query: str)
BaseOutputParser.format_messages(self, messages: List[ChatMessage])
BaseOutputParser.parse(self, output: str)
BasePydanticProgram.__call__(self, *args: Any, **kwds: Any)
BasePydanticProgram.output_cls(self)


repos/llama_index/llama-index-legacy/llama_index/legacy/utils.py
-------------------------functions----------------------
_get_colored_text(text: str, color: str)
add_sync_version(func: Any)
concat_dirs(dirname: str, basename: str)
count_tokens(text: str)
get_cache_dir()
get_color_mapping(items: List[str], use_llama_index_colors: bool  =  True)
get_new_id(d: Set)
get_new_int_id(d: Set)
get_tokenizer()
get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str)
get_transformer_tokenizer_fn(model_name: str)
infer_torch_device()
iter_batch(iterable: Union[Iterable, Generator], size: int)
print_text(text: str, color: Optional[str]  =  None, end: str  =  "")
retry_on_exceptions_with_backoff(lambda_fn: Callable, errors_to_retry: List[ErrorToRetry], max_tries: int  =  10, min_backoff_secs: float  =  0.5, max_backoff_secs: float  =  60.0, )
set_global_tokenizer(tokenizer: Union[Tokenizer, Callable[[str], list]])
temp_set_attrs(obj: Any, **kwargs: Any)
truncate_text(text: str, max_length: int)
unit_generator(x: Any)

-------------------------methods----------------------
GlobalsHelper.__init__(self)
GlobalsHelper.stopwords(self)
Tokenizer.encode(self, text: str, *args: Any, **kwargs: Any)


repos/llama_index/llama-index-legacy/tests/__init__.py


repos/llama_index/llama-index-legacy/tests/agent/__init__.py


repos/llama_index/llama-index-legacy/tests/callbacks/__init__.py


repos/llama_index/llama-index-legacy/tests/callbacks/test_llama_debug.py
-------------------------functions----------------------
test_flush_events()
test_get_event_stats()
test_ignore_events()
test_on_event_end()
test_on_event_start()



repos/llama_index/llama-index-legacy/tests/callbacks/test_token_counter.py
-------------------------functions----------------------
test_on_event_end()
test_on_event_start()



repos/llama_index/llama-index-legacy/tests/chat_engine/__init__.py


repos/llama_index/llama-index-legacy/tests/chat_engine/test_condense_plus_context.py
-------------------------functions----------------------
override_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any)
test_condense_plus_context_chat_engine(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-legacy/tests/chat_engine/test_condense_question.py
-------------------------functions----------------------
test_condense_question_chat_engine(mock_service_context: ServiceContext, )
test_condense_question_chat_engine_with_init_history(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-legacy/tests/chat_engine/test_simple.py
-------------------------functions----------------------
test_simple_chat_engine(mock_service_context: ServiceContext, )
test_simple_chat_engine_with_init_history(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-legacy/tests/conftest.py
-------------------------functions----------------------
allow_networking(monkeypatch: pytest.MonkeyPatch)
mock_llm()
mock_openai_credentials()
mock_service_context(patch_token_text_splitter: Any, patch_llm_predictor: Any, )
patch_llm_predictor(monkeypatch: pytest.MonkeyPatch)
patch_token_text_splitter(monkeypatch: pytest.MonkeyPatch)
pytest_addoption(parser: pytest.Parser)
pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item])
pytest_configure(config: pytest.Config)

-------------------------methods----------------------
CachedOpenAIApiKeys.__enter__(self)
CachedOpenAIApiKeys.__exit__(self, *exc: object)
CachedOpenAIApiKeys.__init__(self, set_env_key_to: Optional[str]  =  "", set_library_key_to: Optional[str]  =  None, set_fake_key: bool  =  False, set_env_type_to: Optional[str]  =  "", set_library_type_to: str  =  "open_ai", # default value in openai package)


repos/llama_index/llama-index-legacy/tests/embeddings/__init__.py


repos/llama_index/llama-index-legacy/tests/embeddings/test_azure_openai.py
-------------------------functions----------------------
test_custom_http_client(azure_openai_mock: MagicMock)



repos/llama_index/llama-index-legacy/tests/embeddings/test_base.py
-------------------------functions----------------------
mock_get_text_embedding(text: str)
mock_get_text_embeddings(texts: List[str])
test_embedding_similarity()
test_embedding_similarity_euclidean()
test_get_text_embeddings(_mock_get_text_embeddings: Any, _mock_get_text_embedding: Any)
test_mean_agg()
test_validates_api_key_is_present()



repos/llama_index/llama-index-legacy/tests/embeddings/test_bedrock.py
-------------------------methods----------------------
TestBedrockEmbedding.test_get_text_embedding_cohere(self)
TestBedrockEmbedding.test_get_text_embedding_titan(self)


repos/llama_index/llama-index-legacy/tests/embeddings/test_elasticsearch.py
-------------------------functions----------------------
es_password()
es_url()
es_username()
model_id()
test_elasticsearch_embedding_constructor(model_id: str, es_url: str, es_username: str, es_password: str)



repos/llama_index/llama-index-legacy/tests/embeddings/test_fastembed.py
-------------------------functions----------------------
test_fastembed_embedding_texts_batch(model_name: str, max_length: int, doc_embed_type: Literal["default", "passage"], threads: int, )
test_fastembed_query_embedding(model_name: str, max_length: int)



repos/llama_index/llama-index-legacy/tests/embeddings/test_gradient.py
-------------------------functions----------------------
gradient_access_token()
gradient_host()
gradient_model_slug()
gradient_workspace_id()
test_gradientai_can_receive_multiple_text_embeddings(gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str)
test_gradientai_can_receive_query_embedding(gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str)
test_gradientai_can_receive_text_embedding(gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str)
test_gradientai_cannot_support_batches_larger_than_100(gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str)
test_gradientai_embedding_constructor(gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str)
test_gradientai_throws_if_not_installed(gradient_access_token: str, gradient_model_slug: str, gradient_workspace_id: str)
test_gradientai_throws_without_proper_auth(gradient_model_slug: str, gradient_workspace_id: str)



repos/llama_index/llama-index-legacy/tests/embeddings/test_huggingface.py
-------------------------functions----------------------
fixture_hf_inference_api_embedding()

-------------------------methods----------------------
TestHuggingFaceInferenceAPIEmbeddings.test_class_name(self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding)
TestHuggingFaceInferenceAPIEmbeddings.test_embed_query(self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding)
TestHuggingFaceInferenceAPIEmbeddings.test_embed_query_one_dimension(self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding)
TestHuggingFaceInferenceAPIEmbeddings.test_serialization(self, hf_inference_api_embedding: HuggingFaceInferenceAPIEmbedding)
TestHuggingFaceInferenceAPIEmbeddings.test_using_recommended_model(self)


repos/llama_index/llama-index-legacy/tests/embeddings/test_llm_rails.py
-------------------------functions----------------------
api_key()
model_id()
test_llm_rails_embedding_constructor(model_id: str, api_key: str)



repos/llama_index/llama-index-legacy/tests/embeddings/test_utils.py
-------------------------functions----------------------
mock_hf_embeddings(*args: Any, **kwargs: Dict[str, Any])
mock_openai_embeddings(*args: Any, **kwargs: Dict[str, Any])
test_resolve_embed_model(monkeypatch: MonkeyPatch)



repos/llama_index/llama-index-legacy/tests/evaluation/test_base.py
-------------------------functions----------------------
test_evaluator_basic()

-------------------------methods----------------------
MockEvaluator.__init__(self, mock_score: float  =  1.0, mock_passing: bool  =  True, mock_feedback: str  =  "test feedback", )
MockEvaluator._get_prompts(self)
MockEvaluator._update_prompts(self, prompts: PromptDictType)


repos/llama_index/llama-index-legacy/tests/evaluation/test_dataset_generation.py
-------------------------functions----------------------
test_dataset_generation(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-legacy/tests/extractors/test_metadata_extractor.py
-------------------------functions----------------------
test_metadata_extractor()



repos/llama_index/llama-index-legacy/tests/finetuning/__init__.py


repos/llama_index/llama-index-legacy/tests/finetuning/test_base.py


repos/llama_index/llama-index-legacy/tests/indices/__init__.py


repos/llama_index/llama-index-legacy/tests/indices/conftest.py
-------------------------functions----------------------
documents()
nodes()



repos/llama_index/llama-index-legacy/tests/indices/test_loading.py
-------------------------functions----------------------
test_load_index_from_storage_faiss_vector_store(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext, )
test_load_index_from_storage_multiple(nodes: List[BaseNode], tmp_path: Path, mock_service_context: ServiceContext, )
test_load_index_from_storage_retrieval_result_identical(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext, )
test_load_index_from_storage_simple(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext)
test_load_index_query_engine_service_context(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext, )



repos/llama_index/llama-index-legacy/tests/indices/test_loading_graph.py
-------------------------functions----------------------
test_load_graph_from_storage_simple(documents: List[Document], tmp_path: Path, mock_service_context: ServiceContext, )



repos/llama_index/llama-index-legacy/tests/indices/test_prompt_helper.py
-------------------------functions----------------------
test_get_biggest_prompt()
test_get_chunk_size(prompt: str, chunk_size_limit: Optional[int], num_chunks: int, padding: int, expected: Union[int, Type[Exception]], )
test_get_numbered_text_from_nodes()
test_get_text_splitter()
test_get_text_splitter_partial()
test_repack()
test_truncate()



repos/llama_index/llama-index-legacy/tests/indices/test_service_context.py
-------------------------functions----------------------
test_service_context_serialize()



repos/llama_index/llama-index-legacy/tests/indices/test_utils.py
-------------------------functions----------------------
test_expand_tokens_with_subtokens()



repos/llama_index/llama-index-legacy/tests/ingestion/test_cache.py
-------------------------functions----------------------
test_cache()
test_cache_clear()

-------------------------methods----------------------
DummyTransform.__call__(self, nodes: List[BaseNode], **kwargs: Any)


repos/llama_index/llama-index-legacy/tests/ingestion/test_pipeline.py
-------------------------functions----------------------
mock_hf_embeddings(*args: Any, **kwargs: Dict[str, Any])
mock_openai_embeddings(*args: Any, **kwargs: Dict[str, Any])
test_resolve_embed_model(monkeypatch: MonkeyPatch)



repos/llama_index/llama-index-legacy/tests/langchain_helpers/__init__.py


repos/llama_index/llama-index-legacy/tests/llm_predictor/__init__.py


repos/llama_index/llama-index-legacy/tests/llm_predictor/test_base.py
-------------------------functions----------------------
mock_llmpredictor_predict(prompt: BasePromptTemplate, **prompt_args: Any)
test_struct_llm_predictor(mock_init: Any, mock_predict: Any)

-------------------------methods----------------------
MockOutputParser.format(self, output: str)
MockOutputParser.parse(self, output: str)


repos/llama_index/llama-index-legacy/tests/llms/__init__.py


repos/llama_index/llama-index-legacy/tests/llms/test_ai21.py
-------------------------functions----------------------
mock_chat(*args: Any, **kwargs: Any)
mock_completion(*args: Any, **kwargs: Any)
test_completion_model_basic(monkeypatch: MonkeyPatch)



repos/llama_index/llama-index-legacy/tests/llms/test_anthropic.py
-------------------------functions----------------------
test_basic()
test_streaming()



repos/llama_index/llama-index-legacy/tests/llms/test_anthropic_utils.py
-------------------------functions----------------------
test_anthropic_modelname_to_contextsize()
test_messages_to_anthropic_prompt()



repos/llama_index/llama-index-legacy/tests/llms/test_azure_openai.py
-------------------------functions----------------------
test_custom_http_client(sync_azure_openai_mock: MagicMock)



repos/llama_index/llama-index-legacy/tests/llms/test_bedrock.py
-------------------------functions----------------------
get_invoke_model_response(payload: str)
test_model_basic(model: str, complete_request: str, response_body: str, chat_request: str)
test_model_streaming(monkeypatch: MonkeyPatch)

-------------------------methods----------------------
MockEventStream.__iter__(self)
MockStreamCompletionWithRetry.__init__(self, expected_prompt: str)
MockStreamCompletionWithRetry.mock_stream_completion_with_retry(self, request_body: str, *args: Any, **kwargs: Any)


repos/llama_index/llama-index-legacy/tests/llms/test_cohere.py
-------------------------functions----------------------
mock_chat_with_retry(*args: Any, **kwargs: Any)
mock_completion_with_retry(*args: Any, **kwargs: Any)
test_completion_model_basic(monkeypatch: MonkeyPatch)



repos/llama_index/llama-index-legacy/tests/llms/test_custom.py
-------------------------functions----------------------
test_basic()
test_streaming()

-------------------------methods----------------------
TestLLM.__init__(self)
TestLLM.complete(self, prompt: str, formatted: bool  =  False, **kwargs: Any)
TestLLM.metadata(self)
TestLLM.stream_complete(self, prompt: str, formatted: bool  =  False, **kwargs: Any)


repos/llama_index/llama-index-legacy/tests/llms/test_gemini.py
-------------------------functions----------------------
test_gemini()
test_gemini_stream()

-------------------------methods----------------------
FakeGoogleDataclass.__init__(self, d: Mapping[str, Any], *args: Any, **kwargs: Any)
FakeGoogleDataclass.to_dict(self)
MockGenaiPackage.GenerativeModel(self, **kwargs: Any)
MockGenaiPackage._gen_content(self, contents: Any, *, stream: bool  =  False, **kwargs: Any)
MockGenaiPackage.get_model(self, name: str, **kwargs: Any)


repos/llama_index/llama-index-legacy/tests/llms/test_gradient.py
-------------------------functions----------------------
test_gradient_adapter()
test_gradient_base()

-------------------------methods----------------------
GradientModel.complete(self, query: str, max_generated_token_count: int)
MockGradient.close(self)
MockGradient.get_base_model(self, base_model_slug: str)
MockGradient.get_model_adapter(self, model_adapter_id: str)


repos/llama_index/llama-index-legacy/tests/llms/test_huggingface.py
-------------------------functions----------------------
fixture_hf_inference_api()

-------------------------methods----------------------
TestHuggingFaceInferenceAPI.test_chat(self, hf_inference_api: HuggingFaceInferenceAPI)
TestHuggingFaceInferenceAPI.test_chat_text_generation(self, hf_inference_api: HuggingFaceInferenceAPI)
TestHuggingFaceInferenceAPI.test_class_name(self, hf_inference_api: HuggingFaceInferenceAPI)
TestHuggingFaceInferenceAPI.test_complete(self, hf_inference_api: HuggingFaceInferenceAPI)
TestHuggingFaceInferenceAPI.test_instantiation(self)


repos/llama_index/llama-index-legacy/tests/llms/test_konko.py
-------------------------functions----------------------
teardown_module()
test_chat_model_basic_non_openai_model()
test_chat_model_basic_openai_model()
test_chat_model_streaming()



repos/llama_index/llama-index-legacy/tests/llms/test_langchain.py
-------------------------functions----------------------
test_basic()
test_from_lc_messages()
test_metadata_sets_model_name()
test_to_lc_messages()



repos/llama_index/llama-index-legacy/tests/llms/test_litellm.py
-------------------------functions----------------------
mock_chat_completion(*args: Any, **kwargs: Any)
mock_chat_completion_stream(*args: Any, **kwargs: Any)
mock_completion(*args: Any, **kwargs: Any)
mock_completion_stream(*args: Any, **kwargs: Any)
test_chat_model_basic(monkeypatch: MonkeyPatch)
test_deep_infra()
test_metadata()
test_openai()
test_tg_ai()



repos/llama_index/llama-index-legacy/tests/llms/test_llama_utils.py
-------------------------functions----------------------
chat_messages_assistant_first()
chat_messages_first_chat()
chat_messages_first_chat_no_system(chat_messages_first_chat: Sequence[ChatMessage], )
chat_messages_second_chat()
chat_messages_second_chat_no_system(chat_messages_second_chat: Sequence[ChatMessage], )
chat_messages_third_chat()
chat_messages_third_chat_no_system(chat_messages_third_chat: Sequence[ChatMessage], )
chat_messages_user_twice()
test_completion_to_prompt()
test_completion_to_prompt_default()
test_error_assistant_first(chat_messages_assistant_first: Sequence[ChatMessage], )
test_error_user_twice(chat_messages_user_twice: Sequence[ChatMessage])
test_first_chat(chat_messages_first_chat: Sequence[ChatMessage])
test_first_chat_default(chat_messages_first_chat_no_system: Sequence[ChatMessage], )
test_second_chat(chat_messages_second_chat: Sequence[ChatMessage])
test_second_chat_default(chat_messages_second_chat_no_system: Sequence[ChatMessage], )
test_third_chat(chat_messages_third_chat: Sequence[ChatMessage])
test_third_chat_default(chat_messages_third_chat_no_system: Sequence[ChatMessage], )



repos/llama_index/llama-index-legacy/tests/llms/test_localai.py
-------------------------functions----------------------
mock_chat_completion(text: str)
mock_completion(text: str)
test_chat(MockSyncOpenAI: MagicMock)
test_completion(MockSyncOpenAI: MagicMock)
test_interfaces()
test_serialization()



repos/llama_index/llama-index-legacy/tests/llms/test_openai.py
-------------------------functions----------------------
mock_chat_completion(*args: Any, **kwargs: Any)
mock_chat_completion_stream(*args: Any, **kwargs: Any)
mock_chat_completion_stream_v1(*args: Any, **kwargs: Any)
mock_chat_completion_v1(*args: Any, **kwargs: Any)
mock_completion(*args: Any, **kwargs: Any)
mock_completion_stream(*args: Any, **kwargs: Any)
mock_completion_stream_v1(*args: Any, **kwargs: Any)
mock_completion_v1(*args: Any, **kwargs: Any)
test_chat_model_basic(MockSyncOpenAI: MagicMock)
test_chat_model_streaming(MockSyncOpenAI: MagicMock)
test_completion_model_basic(MockSyncOpenAI: MagicMock)
test_completion_model_streaming(MockSyncOpenAI: MagicMock)
test_validates_api_key_is_present()



repos/llama_index/llama-index-legacy/tests/llms/test_openai_like.py
-------------------------functions----------------------
mock_chat_completion(text: str)
mock_completion(text: str)
test_chat(MockSyncOpenAI: MagicMock)
test_completion(MockSyncOpenAI: MagicMock)
test_interfaces()
test_serialization()

-------------------------methods----------------------
StubTokenizer.encode(self, text: str)


repos/llama_index/llama-index-legacy/tests/llms/test_openai_utils.py
-------------------------functions----------------------
azure_chat_messages_with_function_calling()
azure_openai_message_dicts_with_function_calling()
chat_messages_with_function_calling()
openi_message_dicts_with_function_calling()
test_from_openai_message_dicts_function_calling(openi_message_dicts_with_function_calling: List[ChatCompletionMessageParam], chat_messages_with_function_calling: List[ChatMessage], )
test_from_openai_messages_function_calling_azure(azure_openai_message_dicts_with_function_calling: List[ChatCompletionMessage], azure_chat_messages_with_function_calling: List[ChatMessage], )
test_to_openai_message_dicts_basic_enum()
test_to_openai_message_dicts_basic_string()
test_to_openai_message_dicts_function_calling(chat_messages_with_function_calling: List[ChatMessage], openi_message_dicts_with_function_calling: List[ChatCompletionMessageParam], )
test_to_openai_message_with_pydantic_description()
test_to_openai_tool_with_provided_description()



repos/llama_index/llama-index-legacy/tests/llms/test_palm.py
-------------------------functions----------------------
_mock_palm_completion(model_name: str, prompt: str, **kwargs: Any)
test_palm()

-------------------------methods----------------------
MockPalmPackage._mock_models(self)
MockPalmPackage.generate_text(self, model: str, prompt: str, **kwargs: Any)
MockPalmPackage.list_models(self)


repos/llama_index/llama-index-legacy/tests/llms/test_rungpt.py
-------------------------functions----------------------
mock_chat_completion(*args: Any, **kwargs: Any)
mock_chat_completion_stream(*args: Any, **kwargs: Any)
mock_chat_history(*args: Any, **kwargs: Any)
mock_completion(*args: Any, **kwargs: Any)
mock_completion_stream(*args: Any, **kwargs: Any)
test_chat(chat_history: List[ChatMessage])
test_complete()
test_init()
test_stream_chat(chat_history: List[ChatMessage])
test_stream_complete()



repos/llama_index/llama-index-legacy/tests/llms/test_vertex.py
-------------------------functions----------------------
test_vertex_call()
test_vertex_generate()
test_vertex_generate_code()
test_vertex_initialization()
test_vertex_stream()



repos/llama_index/llama-index-legacy/tests/llms/test_vllm.py
-------------------------functions----------------------
test_vllm_call()
test_vllm_initialization()



repos/llama_index/llama-index-legacy/tests/llms/test_watsonx.py
-------------------------functions----------------------
test_model_basic()
test_model_streaming()

-------------------------methods----------------------
MockStreamResponse.__iter__(self)


repos/llama_index/llama-index-legacy/tests/llms/test_xinference.py
-------------------------functions----------------------
mock_chat_stream_iterator()
test_chat(chat_history: Sequence[ChatMessage])
test_complete()
test_init()
test_stream_chat(chat_history: Sequence[ChatMessage])
test_stream_complete()

-------------------------methods----------------------
MockRESTfulClient.get_model(self)
MockXinference.load_model(self, model_uid: str, endpoint: str, )
MockXinferenceModel.chat(self, prompt: str, chat_history: List[Mapping[str, Any]], generate_config: Dict[str, Any], )


repos/llama_index/llama-index-legacy/tests/logger/__init__.py


repos/llama_index/llama-index-legacy/tests/logger/test_base.py
-------------------------functions----------------------
test_logger()
test_logger_metadata()



repos/llama_index/llama-index-legacy/tests/memory/test_chat_memory_buffer.py
-------------------------functions----------------------
test_dict_save_load()
test_get_when_initial_tokens_exceed_limit_raises_value_error()
test_get_when_initial_tokens_less_than_limit_returns_history()
test_get_when_initial_tokens_same_as_limit_removes_message()
test_get_when_space_for_all_but_first_message_removes_first_message_and_answer() -> (None)
test_get_when_space_for_assistant_message_removes_assistant_message_at_start_of_history() -> (None)
test_get_when_space_for_second_message_and_answer_removes_only_first_message_and_answer() -> (None)
test_max_tokens()
test_pickle()
test_put_get()
test_set()
test_sting_save_load()



repos/llama_index/llama-index-legacy/tests/mock_utils/__init__.py


repos/llama_index/llama-index-legacy/tests/mock_utils/mock_predict.py
-------------------------functions----------------------
_mock_answer(prompt_args: Dict)
_mock_choice_select(prompt_args: Dict)
_mock_conversation(prompt_args: Dict)
_mock_decompose_query(prompt_args: Dict)
_mock_input(prompt_args: Dict)
_mock_insert_predict()
_mock_keyword_extract(prompt_args: Dict)
_mock_kg_triplet_extract(prompt_args: Dict)
_mock_multi_select(prompt_args: Dict)
_mock_pandas(prompt_args: Dict)
_mock_query_keyword_extract(prompt_args: Dict)
_mock_query_select()
_mock_refine(prompt_args: Dict)
_mock_schema_extract(prompt_args: Dict)
_mock_single_select()
_mock_sql_response_synthesis(prompt_args: Dict)
_mock_sql_response_synthesis_v2(prompt_args: Dict)
_mock_sub_questions()
_mock_summary_predict(prompt_args: Dict)
_mock_text_to_sql(prompt_args: Dict)
mock_llmpredictor_predict(prompt: BasePromptTemplate, **prompt_args: Any)
patch_llmpredictor_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any)



repos/llama_index/llama-index-legacy/tests/mock_utils/mock_prompts.py


repos/llama_index/llama-index-legacy/tests/mock_utils/mock_text_splitter.py
-------------------------functions----------------------
mock_token_splitter_newline(text: str, metadata_str: Optional[str]  =  None)
patch_token_splitter_newline(self: Any, text: str, metadata_str: Optional[str]  =  None)



repos/llama_index/llama-index-legacy/tests/mock_utils/mock_utils.py
-------------------------functions----------------------
mock_extract_keywords(text_chunk: str, max_keywords: Optional[int]  =  None, filter_stopwords: bool  =  True)
mock_tokenizer(text: str)



repos/llama_index/llama-index-legacy/tests/multi_modal_llms/__init__.py


repos/llama_index/llama-index-legacy/tests/multi_modal_llms/test_replicate_multi_modal.py
-------------------------functions----------------------
mock_completion(*args: Any, **kwargs: Any)
test_completion_model_basic(monkeypatch: MonkeyPatch)



repos/llama_index/llama-index-legacy/tests/node_parser/metadata_extractor.py
-------------------------functions----------------------
test_metadata_extractor(mock_service_context: ServiceContext)



repos/llama_index/llama-index-legacy/tests/node_parser/sentence_window.py
-------------------------functions----------------------
test_split_and_window()



repos/llama_index/llama-index-legacy/tests/node_parser/test_html.py
-------------------------functions----------------------
test_multiple_tags_splits()
test_neighbor_tags_splits()
test_nesting_tags_splits()
test_no_splits()
test_single_splits()



repos/llama_index/llama-index-legacy/tests/node_parser/test_json.py
-------------------------functions----------------------
test_split_empty_text()
test_split_invalid_json()
test_split_valid_dict_json()
test_split_valid_json()
test_split_valid_json_defaults()



repos/llama_index/llama-index-legacy/tests/node_parser/test_markdown.py
-------------------------functions----------------------
test_header_metadata()
test_header_splits()
test_non_header_splits()
test_pre_header_content()



repos/llama_index/llama-index-legacy/tests/node_parser/test_markdown_element.py
-------------------------functions----------------------
test_complex_md()
test_llama2_bad_md()
test_md_table_extraction()
test_md_table_extraction_broken_table()



repos/llama_index/llama-index-legacy/tests/node_parser/test_semantic_splitter.py
-------------------------functions----------------------
test_grouped_semantically()
test_split_and_permutated()



repos/llama_index/llama-index-legacy/tests/node_parser/test_unstructured.py
-------------------------functions----------------------
test_html_table_extraction()



repos/llama_index/llama-index-legacy/tests/objects/__init__.py


repos/llama_index/llama-index-legacy/tests/objects/test_base.py
-------------------------functions----------------------
test_object_index(mock_service_context: ServiceContext)
test_object_index_persist(mock_service_context: ServiceContext)
test_object_index_with_tools(mock_service_context: ServiceContext)



repos/llama_index/llama-index-legacy/tests/objects/test_node_mapping.py
-------------------------functions----------------------
test_simple_object_node_mapping()
test_simple_object_node_mapping_persist()
test_sql_table_node_mapping_to_node(mocker: MockerFixture)
test_tool_object_node_mapping()

-------------------------methods----------------------
TestObject.__hash__(self)
TestObject.__str__(self)
TestSQLDatabase.__init__(self)


repos/llama_index/llama-index-legacy/tests/output_parsers/__init__.py


repos/llama_index/llama-index-legacy/tests/output_parsers/test_base.py
-------------------------functions----------------------
test_lc_output_parser()



repos/llama_index/llama-index-legacy/tests/output_parsers/test_pydantic.py
-------------------------functions----------------------
test_pydantic()
test_pydantic_format()



repos/llama_index/llama-index-legacy/tests/output_parsers/test_selection.py
-------------------------functions----------------------
output_parser()
test_format(output_parser: SelectionOutputParser)



repos/llama_index/llama-index-legacy/tests/output_parsers/test_utils.py
-------------------------functions----------------------
test_extract_json_str()



repos/llama_index/llama-index-legacy/tests/param_tuner/__init__.py


repos/llama_index/llama-index-legacy/tests/param_tuner/test_base.py
-------------------------functions----------------------
_mock_obj_function(param_dict: Dict)
test_param_tuner()



repos/llama_index/llama-index-legacy/tests/playground/__init__.py


repos/llama_index/llama-index-legacy/tests/playground/test_base.py
-------------------------functions----------------------
test_from_docs(mock_service_context: ServiceContext, )
test_get_set_compare(mock_service_context: ServiceContext, )
test_validation()

-------------------------methods----------------------
MockEmbedding._get_query_embedding(self, query: str)
MockEmbedding._get_text_embedding(self, text: str)
MockEmbedding.class_name(cls)


repos/llama_index/llama-index-legacy/tests/postprocessor/__init__.py


repos/llama_index/llama-index-legacy/tests/postprocessor/test_base.py
-------------------------functions----------------------
test_embedding_recency_postprocessor(mock_service_context: ServiceContext, )
test_fixed_recency_postprocessor(mock_service_context: ServiceContext, )
test_forward_back_processor(tmp_path: Path)
test_keyword_postprocessor()
test_keyword_postprocessor_for_non_english()
test_time_weighted_postprocessor()



repos/llama_index/llama-index-legacy/tests/postprocessor/test_llm_rerank.py
-------------------------functions----------------------
mock_format_node_batch_fn(nodes: List[BaseNode])
mock_llmpredictor_predict(self: Any, prompt: BasePromptTemplate, **prompt_args: Any)
test_llm_rerank(mock_service_context: ServiceContext)



repos/llama_index/llama-index-legacy/tests/postprocessor/test_longcontext_reorder.py
-------------------------functions----------------------
test_long_context_reorder()



repos/llama_index/llama-index-legacy/tests/postprocessor/test_metadata_replacement.py
-------------------------functions----------------------
test_metadata_replacement()



repos/llama_index/llama-index-legacy/tests/postprocessor/test_optimizer.py
-------------------------functions----------------------
mock_get_text_embedding(text: str)
mock_get_text_embedding_chinese(text: str)
mock_get_text_embeddings(texts: List[str])
mock_get_text_embeddings_chinese(texts: List[str])
mock_tokenizer_fn(text: str)
mock_tokenizer_fn2(text: str)
test_optimizer(_mock_embeds: Any, _mock_embed: Any)



repos/llama_index/llama-index-legacy/tests/program/__init__.py


repos/llama_index/llama-index-legacy/tests/program/test_guidance.py
-------------------------functions----------------------
test_guidance_pydantic_program()



repos/llama_index/llama-index-legacy/tests/program/test_llm_program.py
-------------------------functions----------------------
test_llm_program()
test_llm_program_with_messages()
test_llm_program_with_messages_and_chat()

-------------------------methods----------------------
MockChatLLM.chat(self, prompt: str)
MockChatLLM.metadata(self)
MockLLM.complete(self, prompt: str)
MockLLM.metadata(self)


repos/llama_index/llama-index-legacy/tests/program/test_lmformatenforcer.py
-------------------------functions----------------------
test_lmformatenforcer_pydantic_program()



repos/llama_index/llama-index-legacy/tests/program/test_multi_modal_llm_program.py
-------------------------functions----------------------
test_multi_modal_llm_program()

-------------------------methods----------------------
MockMultiModalLLM.complete(self, prompt: str, image_documents: Sequence[ImageDocument])
MockMultiModalLLM.metadata(self)


repos/llama_index/llama-index-legacy/tests/prompts/__init__.py


repos/llama_index/llama-index-legacy/tests/prompts/test_base.py
-------------------------functions----------------------
output_parser()
test_chat_template()
test_chat_template_output_parser(output_parser: BaseOutputParser)
test_function_mappings()
test_langchain_selector_template()
test_langchain_template()
test_selector_template()
test_template()
test_template_output_parser(output_parser: BaseOutputParser)
test_template_var_mappings()

-------------------------methods----------------------
MockOutputParser.__init__(self, format_string: str)
MockOutputParser.format(self, query: str)
MockOutputParser.parse(self, output: str)


repos/llama_index/llama-index-legacy/tests/prompts/test_guidance_utils.py
-------------------------functions----------------------
test_convert_pydantic_to_guidance_output_template_nested()
test_convert_pydantic_to_guidance_output_template_simple()
test_convert_to_handlebars()



repos/llama_index/llama-index-legacy/tests/prompts/test_mixin.py
-------------------------functions----------------------
test_prompt_mixin()

-------------------------methods----------------------
MockObject1.__init__(self)
MockObject1._get_prompt_modules(self)
MockObject1._get_prompts(self)
MockObject1._update_prompts(self, prompts: PromptDictType)
MockObject2.__init__(self)
MockObject2._get_prompt_modules(self)
MockObject2._get_prompts(self)
MockObject2._update_prompts(self, prompts: PromptDictType)


repos/llama_index/llama-index-legacy/tests/prompts/test_utils.py
-------------------------functions----------------------
test_get_template_vars()



repos/llama_index/llama-index-legacy/tests/query_engine/test_cogniswitch_query_engine.py
-------------------------functions----------------------
query_engine()
test_query_knowledge_successful(mock_post: Any, query_engine: CogniswitchQueryEngine)
test_query_knowledge_unsuccessful(mock_post: Any, query_engine: CogniswitchQueryEngine)



repos/llama_index/llama-index-legacy/tests/query_engine/test_pandas.py
-------------------------functions----------------------
test_pandas_query_engine(mock_service_context: ServiceContext)



repos/llama_index/llama-index-legacy/tests/query_engine/test_retriever_query_engine.py
-------------------------functions----------------------
test_query_engine_falls_back_to_inheriting_retrievers_service_context()



repos/llama_index/llama-index-legacy/tests/query_pipeline/__init__.py


repos/llama_index/llama-index-legacy/tests/query_pipeline/test_components.py
-------------------------functions----------------------
bar_fn(a: Any, b: Any)
foo_fn(a: int, b: int  =  1, c: int  =  2)
sum_fn(a: List[int])
test_arg_component()
test_fn_components()
test_fn_pipeline()
test_kwarg_component()
test_selector_component()

-------------------------methods----------------------
MockSelector._get_prompts(self)
MockSelector._select(self, choices: Sequence[ToolMetadata], query: QueryBundle)
MockSelector._update_prompts()


repos/llama_index/llama-index-legacy/tests/query_pipeline/test_query.py
-------------------------functions----------------------
test_query_pipeline_chain()
test_query_pipeline_chain_str()
test_query_pipeline_conditional_edges()
test_query_pipeline_init()
test_query_pipeline_input_component()
test_query_pipeline_multi()
test_query_pipeline_partial()
test_query_pipeline_single_arg_inp()
test_query_pipeline_sub()

-------------------------methods----------------------
Chainable2._as_query_component(self, **kwargs: Any)
QueryComponent1._run_component(self, **kwargs: Any)
QueryComponent1._validate_component_inputs(self, input: Dict[str, Any])
QueryComponent1.input_keys(self)
QueryComponent1.output_keys(self)
QueryComponent1.set_callback_manager(self, callback_manager: Any)
QueryComponent2._run_component(self, **kwargs: Any)
QueryComponent2._validate_component_inputs(self, input: Dict[str, Any])
QueryComponent2.input_keys(self)
QueryComponent2.output_keys(self)
QueryComponent2.set_callback_manager(self, callback_manager: Any)
QueryComponent3._run_component(self, **kwargs: Any)
QueryComponent3._validate_component_inputs(self, input: Dict[str, Any])
QueryComponent3.input_keys(self)
QueryComponent3.output_keys(self)
QueryComponent3.set_callback_manager(self, callback_manager: Any)


repos/llama_index/llama-index-legacy/tests/question_gen/__init__.py


repos/llama_index/llama-index-legacy/tests/question_gen/test_guidance_generator.py
-------------------------functions----------------------
test_guidance_question_generator()



repos/llama_index/llama-index-legacy/tests/question_gen/test_llm_generators.py
-------------------------functions----------------------
test_llm_question_gen(mock_service_context: ServiceContext, )



repos/llama_index/llama-index-legacy/tests/readers/__init__.py


repos/llama_index/llama-index-legacy/tests/readers/test_file.py
-------------------------functions----------------------
test_error_if_not_dir_or_file()
test_exclude_hidden()
test_excluded_files()
test_file_metadata()
test_filename_as_doc_id()
test_nonrecursive()
test_num_files_limit()
test_parallel_load()
test_recursive()
test_required_exts()
test_specifying_encoding()



repos/llama_index/llama-index-legacy/tests/readers/test_html_reader.py
-------------------------functions----------------------
html_str()



repos/llama_index/llama-index-legacy/tests/readers/test_jaguar.py
-------------------------methods----------------------
TestJaguarReader.setup_class(cls)
TestJaguarReader.teardown_class(cls)
TestJaguarReader.test_add_texts(self)
TestJaguarReader.test_clear(self)
TestJaguarReader.test_create(self)
TestJaguarReader.test_drop(self)
TestJaguarReader.test_login(self)
TestJaguarReader.test_logout(self)
TestJaguarReader.test_query_data_filter(self)
TestJaguarReader.test_query_data_limit(self)
TestJaguarReader.test_query_embedding(self)


repos/llama_index/llama-index-legacy/tests/readers/test_json.py
-------------------------functions----------------------
test_basic()
test_collapse_length()
test_jsonl()
test_levels_back0()



repos/llama_index/llama-index-legacy/tests/readers/test_load_reader.py
-------------------------functions----------------------
test_loading_readers()



repos/llama_index/llama-index-legacy/tests/readers/test_mongo.py
-------------------------functions----------------------
test_load_data()
test_load_data_with_field_name()
test_load_data_with_max_docs()
test_load_data_with_metadata_name()



repos/llama_index/llama-index-legacy/tests/readers/test_simplewebreader.py
-------------------------functions----------------------
test_error_40x()
test_url_metadata()



repos/llama_index/llama-index-legacy/tests/readers/test_string_iterable.py
-------------------------functions----------------------
test_load()



repos/llama_index/llama-index-legacy/tests/response_synthesizers/test_google.py
-------------------------functions----------------------
test_get_response(mock_generate_answer: MagicMock)
test_set_google_config(mock_credentials: MagicMock)
test_synthesize(mock_generate_answer: MagicMock)
test_synthesize_with_max_token_blocking(mock_generate_answer: MagicMock)
test_synthesize_with_recitation_blocking(mock_generate_answer: MagicMock)
test_synthesize_with_safety_blocking(mock_generate_answer: MagicMock)
test_synthesize_with_unknown_blocking(mock_generate_answer: MagicMock)



repos/llama_index/llama-index-legacy/tests/response_synthesizers/test_refine.py
-------------------------functions----------------------
mock_refine_service_context(patch_llm_predictor: Any)
refine_instance(mock_refine_service_context: ServiceContext)
test_constructor_args(mock_refine_service_context: ServiceContext)

-------------------------methods----------------------
MockRefineProgram.__call__(self, *args: Any, context_str: Optional[str]  =  None, context_msg: Optional[str]  =  None, **kwargs: Any)
MockRefineProgram.__init__(self, input_to_query_satisfied: Dict[str, bool])
MockRefineProgram.output_cls(self)


repos/llama_index/llama-index-legacy/tests/retrievers/__init__.py


repos/llama_index/llama-index-legacy/tests/retrievers/test_composable_retriever.py
-------------------------functions----------------------
test_composable_retrieval()



repos/llama_index/llama-index-legacy/tests/selectors/__init__.py


repos/llama_index/llama-index-legacy/tests/selectors/test_llm_selectors.py
-------------------------functions----------------------
test_llm_multi_selector(mock_service_context: ServiceContext, )
test_llm_multi_selector_max_choices(mock_service_context: ServiceContext, )
test_llm_single_selector()



repos/llama_index/llama-index-legacy/tests/storage/__init__.py


repos/llama_index/llama-index-legacy/tests/storage/conftest.py
-------------------------functions----------------------
firestore_kvstore()
mongo_client()
mongo_kvstore(mongo_client: MockMongoClient)
postgres_container()
postgres_kvstore(postgres_container: Dict[str, Union[str, Container]], )
redis_kvstore()
simple_kvstore()



repos/llama_index/llama-index-legacy/tests/storage/test_storage_context.py
-------------------------functions----------------------
test_storage_context_dict()



repos/llama_index/llama-index-legacy/tests/test_exec_utils.py
-------------------------functions----------------------
test_contains_protected_access()



repos/llama_index/llama-index-legacy/tests/test_schema.py
-------------------------functions----------------------
node_with_score(text_node: TextNode)
test_node_with_score_passthrough(node_with_score: NodeWithScore)
test_text_node_hash()
text_node()



repos/llama_index/llama-index-legacy/tests/test_utils.py
-------------------------functions----------------------
fn_with_exception(exception_cls: Optional[Union[Type[Exception], Exception]])
test_get_color_mapping()
test_get_colored_text()
test_iter_batch()
test_print_text(capsys: CaptureFixture)
test_retry_on_conditional_exceptions()
test_retry_on_exceptions_with_backoff()
test_tokenizer()

-------------------------methods----------------------
ConditionalException.__init__(self, should_retry: bool)


repos/llama_index/llama-index-legacy/tests/text_splitter/__init__.py


repos/llama_index/llama-index-legacy/tests/text_splitter/test_code_splitter.py
-------------------------functions----------------------
baz()
baz()
baz()
foo()
foo()
foo()
test__py_custom_parser_code_splitter()
test_cpp_code_splitter()
test_html_code_splitter()
test_python_code_splitter()
test_start_end_char_idx()
test_tsx_code_splitter()
test_typescript_code_splitter()



repos/llama_index/llama-index-legacy/tests/text_splitter/test_sentence_splitter.py
-------------------------functions----------------------
test_chinese_text(chinese_text: str)
test_contiguous_text(contiguous_text: str)
test_edge_case()
test_overlap()
test_paragraphs()
test_sentences()
test_split_texts_multiple()
test_split_texts_singleton()
test_split_texts_with_metadata(english_text: str)
test_split_with_metadata(english_text: str)
test_start_end_char_idx()



repos/llama_index/llama-index-legacy/tests/text_splitter/test_token_splitter.py
-------------------------functions----------------------
test_contiguous_text(contiguous_text: str)
test_split_chinese(chinese_text: str)
test_split_long_token()
test_split_token()
test_split_with_metadata(english_text: str)
test_start_end_char_idx()
test_truncate_token()



repos/llama_index/llama-index-legacy/tests/token_predictor/__init__.py


repos/llama_index/llama-index-legacy/tests/token_predictor/test_base.py
-------------------------functions----------------------
test_token_predictor(mock_split: Any)



repos/llama_index/llama-index-legacy/tests/tools/__init__.py


repos/llama_index/llama-index-legacy/tests/tools/conftest.py
-------------------------functions----------------------
documents()



repos/llama_index/llama-index-legacy/tests/tools/test_base.py
-------------------------functions----------------------
test_function_tool()
test_function_tool_to_langchain()
test_retreiver_tool()
test_tool_fn_schema()
tmp_function(x: int)



repos/llama_index/llama-index-legacy/tests/tools/test_ondemand_loader.py
-------------------------functions----------------------
test_ondemand_loader_tool(tool: OnDemandLoaderTool, )
test_ondemand_loader_tool_langchain(tool: OnDemandLoaderTool, )
tool(mock_service_context: ServiceContext)



repos/llama_index/llama-index-legacy/tests/tools/test_query_engine_tool.py
-------------------------functions----------------------
test_query_engine_tool()

-------------------------methods----------------------
MockQueryEngine.custom_query(self, query_str: str)


repos/llama_index/llama-index-legacy/tests/tools/test_utils.py
-------------------------functions----------------------
test_create_schema_from_function()
test_create_schema_from_function_with_field()



repos/llama_index/llama-index-legacy/tests/utilities/test_sql_wrapper.py
-------------------------functions----------------------
sql_database(request: pytest.FixtureRequest)
test_get_single_table_info(sql_database: SQLDatabase)
test_get_table_columns(sql_database: SQLDatabase)
test_init(sql_database: SQLDatabase)
test_insert_and_run_sql(sql_database: SQLDatabase)
test_long_string_no_truncation(sql_database: SQLDatabase)
test_run_sql_truncation(sql_database: SQLDatabase)



repos/llama_index/llama-index-legacy/tests/vector_stores/__init__.py


repos/llama_index/llama-index-legacy/tests/vector_stores/test_astra.py
-------------------------functions----------------------
astra_db_store()
test_astra_db_create_and_crud(astra_db_store: AstraDBVectorStore)
test_astra_db_queries(astra_db_store: AstraDBVectorStore)



repos/llama_index/llama-index-legacy/tests/vector_stores/test_azureaisearch.py
-------------------------functions----------------------
create_mock_vector_store(search_client: Any, index_name: Optional[str]  =  None, index_management: IndexManagement  =  IndexManagement.NO_VALIDATION, )
create_sample_documents(n: int)
test_azureaisearch_add_one_batch()
test_azureaisearch_add_two_batches()
test_invalid_index_management_for_searchclient()
test_invalid_index_management_for_searchindexclient()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_azurecosmosmongo.py
-------------------------functions----------------------
node_embeddings()

-------------------------methods----------------------
TestAzureMongovCoreVectorSearch.setup(self)
TestAzureMongovCoreVectorSearch.setup_class(cls)
TestAzureMongovCoreVectorSearch.teardown_class(cls)
TestAzureMongovCoreVectorSearch.test_add_and_delete(self)
TestAzureMongovCoreVectorSearch.test_query(self, node_embeddings: List[TextNode])


repos/llama_index/llama-index-legacy/tests/vector_stores/test_cassandra.py
-------------------------methods----------------------
TestCassandraVectorStore.test_cassandra_create_and_crud(self)
TestCassandraVectorStore.test_cassandra_queries(self)


repos/llama_index/llama-index-legacy/tests/vector_stores/test_chromadb.py
-------------------------functions----------------------
node_embeddings()
test_instance_creation_from_collection()
test_instance_creation_from_http_params()
test_instance_creation_from_persist_dir()
vector_store()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_docarray.py
-------------------------functions----------------------
node_embeddings()
test_hnsw(node_embeddings: List[TextNode], tmp_path: Path)
test_hnsw_filters(node_embeddings: List[TextNode], tmp_path: Path)
test_in_memory(node_embeddings: List[TextNode], tmp_path: Path)
test_in_memory_filters(node_embeddings: List[TextNode])



repos/llama_index/llama-index-legacy/tests/vector_stores/test_elasticsearch.py
-------------------------functions----------------------
elasticsearch_connection()
es_store(index_name: str, elasticsearch_connection: Dict)
index_name()
node_embeddings()
test_check_user_agent(index_name: str, node_embeddings: List[TextNode], )
test_instance_creation(index_name: str, elasticsearch_connection: Dict)



repos/llama_index/llama-index-legacy/tests/vector_stores/test_epsilla.py
-------------------------functions----------------------
node_embeddings()
test_add_data_and_query()
test_initiate_store()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_google.py
-------------------------functions----------------------
test_add(mock_get_corpus: MagicMock, mock_get_document: MagicMock, mock_create_document: MagicMock, mock_batch_create_chunks: MagicMock, )
test_class_name()
test_create_corpus(mock_create_corpus: MagicMock)
test_delete(mock_get_corpus: MagicMock, mock_delete_document: MagicMock, )
test_from_corpus(mock_get_corpus: MagicMock)
test_query(mock_get_corpus: MagicMock, mock_query_corpus: MagicMock, )
test_set_google_config(mock_credentials: MagicMock)



repos/llama_index/llama-index-legacy/tests/vector_stores/test_jaguar.py
-------------------------methods----------------------
TestJaguarVectorStore.setup_class(cls)
TestJaguarVectorStore.teardown_class(cls)
TestJaguarVectorStore.test_add_texts(self)
TestJaguarVectorStore.test_clear(self)
TestJaguarVectorStore.test_create(self)
TestJaguarVectorStore.test_drop(self)
TestJaguarVectorStore.test_load_documents_filter(self)
TestJaguarVectorStore.test_login(self)
TestJaguarVectorStore.test_logout(self)
TestJaguarVectorStore.test_query(self)
TestJaguarVectorStore.test_query_cutoff(self)
TestJaguarVectorStore.test_query_filter(self)
TestJaguarVectorStore.test_search_anomalous(self)


repos/llama_index/llama-index-legacy/tests/vector_stores/test_lancedb.py
-------------------------functions----------------------
test_to_llama_similarities_from_df_w_distance()
test_to_llama_similarities_from_df_w_score()
test_to_llama_similarity_from_df_ordinal()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_lantern.py
-------------------------functions----------------------
_get_sample_vector(num: float)
conn()
db(conn: Any)
hybrid_node_embeddings()
index_node_embeddings()
node_embeddings()
pg(db: None)
pg_hybrid(db: None)
test_hybrid_query_fails_if_no_query_str_provided(pg_hybrid: LanternVectorStore, hybrid_node_embeddings: List[TextNode])



repos/llama_index/llama-index-legacy/tests/vector_stores/test_metadata_filters.py
-------------------------functions----------------------
test_legacy_filters()
test_legacy_filters_value_error()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_milvus.py
-------------------------functions----------------------
embedded_milvus()
node_embeddings()
test_add_stores_data(node_embeddings: List[TextNode], embedded_milvus: str)
test_non_default_index_type(node_embeddings: List[TextNode], embedded_milvus: str)
test_search_data(node_embeddings: List[TextNode], embedded_milvus: str)
test_search_data_filter(node_embeddings: List[TextNode], embedded_milvus: str)



repos/llama_index/llama-index-legacy/tests/vector_stores/test_mongodb.py
-------------------------functions----------------------
node_embeddings()

-------------------------methods----------------------
TestMongoDBAtlasVectorSearch.setup(self)
TestMongoDBAtlasVectorSearch.setup_class(cls)
TestMongoDBAtlasVectorSearch.teardown_class(cls)
TestMongoDBAtlasVectorSearch.test_add_and_delete(self)
TestMongoDBAtlasVectorSearch.test_query(self, node_embeddings: List[TextNode])


repos/llama_index/llama-index-legacy/tests/vector_stores/test_pinecone.py
-------------------------functions----------------------
get_version_attr_from_mock_classes(mock_class: Type[Any])
mock_import(name: str, *args: Any, **kwargs: Any)

-------------------------methods----------------------
MockPineconePods.init(api_key: str, environment: str)
MockUnVersionedPineconeRelease.init(api_key: str, environment: str)
TestPineconeVectorStore.setUp(self)
TestPineconeVectorStore.tearDown(self)
TestPineconeVectorStore.test_pods_version(self)
TestPineconeVectorStore.test_serverless_version(self)
TestPineconeVectorStore.test_unversioned_pinecone_client(self)


repos/llama_index/llama-index-legacy/tests/vector_stores/test_postgres.py
-------------------------functions----------------------
_get_sample_vector(num: float)
conn()
db(conn: Any)
hybrid_node_embeddings()
index_node_embeddings()
node_embeddings()
pg(db: None)
pg_hybrid(db: None)
test_hybrid_query_fails_if_no_query_str_provided(pg_hybrid: PGVectorStore, hybrid_node_embeddings: List[TextNode])



repos/llama_index/llama-index-legacy/tests/vector_stores/test_qdrant.py
-------------------------functions----------------------
node_embeddings()
test_add_stores_data(node_embeddings: List[TextNode])
test_add_stores_data_multiple_connections(node_embeddings: List[TextNode])
test_build_query_filter_returns_combined_filter()
test_build_query_filter_returns_empty_filter_on_query_str()
test_build_query_filter_returns_match_any()
test_build_query_filter_returns_none()
test_relative_score_fusion()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_rockset.py
-------------------------functions----------------------
collection_exists(client: Any, collection_name: str  =  "test")
collection_is_empty(client: Any, collection_name: str  =  "test")
test_metadata_filter(vector_store: RocksetVectorStore)
test_query(vector_store: RocksetVectorStore)
vector_store()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_simple.py
-------------------------functions----------------------
_node_embeddings_for_test()

-------------------------methods----------------------
SimpleVectorStoreTest.test_delete_removes_document_from_query_results(self)
SimpleVectorStoreTest.test_query_with_contradictive_filter_returns_no_matches(self)
SimpleVectorStoreTest.test_query_with_exact_filters_returns_single_match(self)
SimpleVectorStoreTest.test_query_with_filter_applies_node_id_filter(self)
SimpleVectorStoreTest.test_query_with_filter_applies_top_k(self)
SimpleVectorStoreTest.test_query_with_filter_on_unknown_field_returns_no_matches(self)
SimpleVectorStoreTest.test_query_with_filters_returns_multiple_matches(self)
SimpleVectorStoreTest.test_query_without_filters_returns_all_rows_sorted_by_similarity(self)


repos/llama_index/llama-index-legacy/tests/vector_stores/test_singlestoredb.py
-------------------------functions----------------------
test_metadata_filter(vector_store: SingleStoreVectorStore)
test_query(vector_store: SingleStoreVectorStore)
vector_store()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_tair.py
-------------------------functions----------------------
get_tair_url()
node_embeddings()
test_add_stores_data(node_embeddings: List[TextNode])
test_delete()
test_query()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_tencentvectordb.py
-------------------------functions----------------------
get_tencent_vdb_store(drop_exists: bool  =  False)
node_embeddings()
test_add_stores_data(node_embeddings: List[TextNode])
test_delete(node_embeddings: List[TextNode])
test_query()
test_query_with_filter(node_embeddings: List[TextNode])



repos/llama_index/llama-index-legacy/tests/vector_stores/test_timescalevector.py
-------------------------functions----------------------
conn()
db(conn: Any)
node_embeddings()
test_add_to_db_query_and_delete(tvs: TimescaleVectorStore, node_embeddings: List[TextNode])
tvs(db: None)
tvs_tp(db: None)



repos/llama_index/llama-index-legacy/tests/vector_stores/test_upstash.py
-------------------------functions----------------------
test_upstash_vector_add(upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode])
test_upstash_vector_query(upstash_vector_store: UpstashVectorStore, text_nodes: List[TextNode])
text_nodes()
upstash_vector_store()



repos/llama_index/llama-index-legacy/tests/vector_stores/test_weaviate.py
-------------------------functions----------------------
test_weaviate_add()



repos/llama_index/llama-index-networks/examples/simple/network_query_engine.py


repos/llama_index/llama-index-networks/examples/simple/query_engine_contributor.py


repos/llama_index/llama-index-networks/examples/simple/retriever_contributor.py


repos/llama_index/llama-index-networks/llama_index/networks/__init__.py


repos/llama_index/llama-index-networks/tests/__init__.py


repos/llama_index/llama-index-networks/tests/network/test_query_engine.py
-------------------------functions----------------------
test_network_query_engine(mock_contributor)



repos/llama_index/llama-index-networks/tests/network/test_retriever.py
-------------------------functions----------------------
test_network_retriever(mock_contributor)



repos/llama_index/llama-index-packs/llama-index-packs-agent-search-retriever/examples/_example.py


repos/llama_index/llama-index-packs/llama-index-packs-agent-search-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-agent-search-retriever/tests/test_packs_agent_search_retriever.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-agents-coa/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-agents-coa/tests/test_packs_agents.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-agents-lats/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-agents-lats/tests/test_packs_lats.py
-------------------------functions----------------------
test_pack()
test_worker()



repos/llama_index/llama-index-packs/llama-index-packs-agents-llm-compiler/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-agents-llm-compiler/tests/test_packs_agents.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-amazon-product-extraction/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-amazon-product-extraction/tests/test_packs_amazon_product_extraction.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-arize-phoenix-query-engine/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-arize-phoenix-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-arize-phoenix-query-engine/tests/test_packs_arize_phoenix_query_engine.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-auto-merging-retriever/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-auto-merging-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-auto-merging-retriever/tests/test_packs_auto_merging_retriever.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-chroma-autoretrieval/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-chroma-autoretrieval/tests/test_packs_chroma_autoretrieval.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-code-hierarchy/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-code-hierarchy/tests/test_code_hierarchy_no_skeleton.py
-------------------------functions----------------------
test_cpp_code_splitter()
test_html_code_splitter()
test_python_code_splitter()
test_python_code_splitter_with_decorators()
test_tsx_code_splitter()
test_typescript_code_splitter()

-------------------------methods----------------------
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()


repos/llama_index/llama-index-packs/llama-index-packs-code-hierarchy/tests/test_code_hierarchy_with_skeleton.py
-------------------------functions----------------------
_handle_extra_radiation_types(datetime_or_doy, epoch_year)
test_html_code_splitter()
test_python_code_splitter()
test_python_code_splitter_with_decorators()
test_typescript_code_splitter()
test_typescript_code_splitter_2()

-------------------------methods----------------------
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()
Foo.bar()


repos/llama_index/llama-index-packs/llama-index-packs-code-hierarchy/tests/test_code_parse_nodes_special_characters.py
-------------------------functions----------------------
function_that_was_cut()
function_that_was_cut()
print_special_character()
test_special_character()



repos/llama_index/llama-index-packs/llama-index-packs-code-hierarchy/tests/test_query_engine.py
-------------------------functions----------------------
code_hierarchy_nodes(request)
print_python(python_text: str)
test_code_splitter_NEXT_relationship_indention(code_hierarchy_nodes: Sequence[BaseNode], )
test_query_by_all_uuids(code_hierarchy_nodes: Sequence[BaseNode])
test_query_by_item_name(name: str, code_hierarchy_nodes: Sequence[BaseNode])
test_query_by_module_name(code_hierarchy_nodes: Sequence[BaseNode])



repos/llama_index/llama-index-packs/llama-index-packs-code-hierarchy/tests/test_utility_methods.py
-------------------------functions----------------------
function()
function()
function()
function()
test_mixed_indentation()
test_mixed_indentation_2()
test_no_indentation()
test_space_indentation()
test_tab_indentation()
test_tab_indentation_2()
test_typescript()
test_typescript_2()



repos/llama_index/llama-index-packs/llama-index-packs-cogniswitch-agent/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-cogniswitch-agent/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-cogniswitch-agent/tests/test_packs_cogniswitch_agent.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-cohere-citation-chat/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-cohere-citation-chat/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-cohere-citation-chat/tests/test_packs_cohere_citation_chat.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-corrective-rag/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-corrective-rag/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-corrective-rag/tests/test_packs_corrective_rag.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-deeplake-deepmemory-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-deeplake-deepmemory-retriever/tests/test_packs_deeplake_deepmemory_retriever.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-deeplake-multimodal-retrieval/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-deeplake-multimodal-retrieval/tests/test_packs_deeplake_multimodal_retrieval.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-dense-x-retrieval/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-dense-x-retrieval/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-dense-x-retrieval/tests/test_packs_dense_x_retrieval.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-diff-private-simple-dataset/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-diff-private-simple-dataset/tests/test_packs_diff_private_examples_gen.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-diff-private-simple-dataset/tests/test_templates.py
-------------------------functions----------------------
test_few_shot_template()



repos/llama_index/llama-index-packs/llama-index-packs-docugami-kg-rag/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-evaluator-benchmarker/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-evaluator-benchmarker/tests/test_packs_evaluator_benchmarker.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-finchat/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-fusion-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-fusion-retriever/tests/test_packs_fusion_retriever.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-fuzzy-citation/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-fuzzy-citation/tests/test_packs_fuzzy_citation.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-gmail-openai-agent/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-gmail-openai-agent/tests/test_packs_gmail_openai_agent.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-gradio-agent-chat/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-gradio-agent-chat/tests/test_packs_gradio_agent_chat.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-gradio-react-agent-chatbot/tests/__init__.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-gradio-react-agent-chatbot/tests/test_packs_gradio_react_agent_chatbot.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-infer-retrieve-rerank/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-infer-retrieve-rerank/tests/test_packs_infer_retrieve_rerank.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-koda-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-koda-retriever/tests/koda_mocking.py
-------------------------methods----------------------
KVMockLLM.class_name(cls)
KVMockLLM.complete(self, prompt: str, **kwargs)
KVMockLLM.random_prompt(self)


repos/llama_index/llama-index-packs/llama-index-packs-koda-retriever/tests/test_koda_retriever.py
-------------------------functions----------------------
test_a_retrieve(setup)
test_categorize(setup)
test_category_retrieve(setup)
test_init(setup)
test_retrieve(setup)



repos/llama_index/llama-index-packs/llama-index-packs-llama-dataset-metadata/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-llama-dataset-metadata/tests/test_packs_llama_dataset_metadata.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-llama-guard-moderator/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-llama-guard-moderator/tests/test_packs_llama_guard_moderator.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-llava-completion/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-llava-completion/tests/test_packs_llava_completion.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-multi-document-agents/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-multi-document-agents/tests/test_packs_multi_document_agents.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-multi-tenancy-rag/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-multi-tenancy-rag/tests/test_packs_multi_tenancy_rag.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-multidoc-autoretrieval/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-multidoc-autoretrieval/tests/test_packs_multidoc_autoretrieval.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-nebulagraph-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-nebulagraph-query-engine/tests/test_packs_nebulagraph_query_engine.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-neo4j-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-neo4j-query-engine/tests/test_packs_neo4j_query_engine.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-node-parser-semantic-chunking/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-node-parser-semantic-chunking/tests/test_packs_node_parser.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-ollama-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-ollama-query-engine/tests/test_packs_ollama_query_engine.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-panel-chatbot/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-panel-chatbot/tests/test_packs_panel_chatbot.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-query-understanding-agent/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-query-understanding-agent/tests/test_packs_query_understanding_agent.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-raft-dataset/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-raft-dataset/tests/test_packs_raft_dataset.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-rag-cli-local/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-rag-cli-local/tests/test_packs_rag_cli_local.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-rag-evaluator/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-rag-evaluator/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-rag-evaluator/tests/test_packs_rag_evaluator.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-rag-fusion-query-pipeline/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-rag-fusion-query-pipeline/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-rag-fusion-query-pipeline/tests/test_packs_query.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-ragatouille-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-ragatouille-retriever/tests/test_packs_ragatouille_retriever.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-raptor/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-raptor/tests/test_packs_raptor.py
-------------------------functions----------------------
test_raptor()



repos/llama_index/llama-index-packs/llama-index-packs-recursive-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-recursive-retriever/tests/test_packs_recursive_retriever.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-redis-ingestion-pipeline/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-redis-ingestion-pipeline/tests/test_packs_redis_ingestion_pipeline.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-resume-screener/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-resume-screener/tests/test_packs_resume_screener.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-retry-engine-weaviate/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-retry-engine-weaviate/tests/test_packs_retry_engine_weaviate.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-searchain/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-searchain/tests/test_packs_searchain.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-self-discover/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-self-discover/tests/test_packs_self_discover.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-self-rag/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-self-rag/tests/test_packs_self_rag.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-sentence-window-retriever/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-sentence-window-retriever/tests/test_packs_sentence_window_retriever.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-snowflake-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-snowflake-query-engine/tests/test_packs_snowflake_query_engine.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-stock-market-data-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-stock-market-data-query-engine/tests/test_packs_stock_market_data_query_engine.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-streamlit-chatbot/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-streamlit-chatbot/tests/test_packs_streamlit_chatbot.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-sub-question-weaviate/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-sub-question-weaviate/tests/test_packs_sub_question_weaviate.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-subdoc-summary/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-tables/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-tables/tests/test_packs_tables.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-trulens-eval-packs/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-trulens-eval-packs/tests/test_packs_trulens_eval_packs.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-vanna/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-vanna/tests/test_packs_vanna.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-vectara-rag/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-vectara-rag/tests/test_packs_vectara_rag.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-voyage-query-engine/examples/example.py


repos/llama_index/llama-index-packs/llama-index-packs-voyage-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-voyage-query-engine/tests/test_packs_voyage_query_engine.py
-------------------------functions----------------------
test_class()



repos/llama_index/llama-index-packs/llama-index-packs-zephyr-query-engine/tests/__init__.py


repos/llama_index/llama-index-packs/llama-index-packs-zephyr-query-engine/tests/test_packs_zephyr_query_engine.py
-------------------------functions----------------------
test_class()


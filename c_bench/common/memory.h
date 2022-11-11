#ifndef __OMPSS_CLUSTER_BENCHMARKS_MEMORY_H__
#define __OMPSS_CLUSTER_BENCHMARKS_MEMORY_H__

/* Memory allocation functions prototypes generators */
#define LMALLOC_FUNC_TYPE_DECL(type) \
type *lmalloc_##type(size_t);

#define LFREE_FUNC_TYPE_DECL(type) \
void lfree_##type(type *, size_t);

#define DMALLOC_FUNC_TYPE_DECL(type) \
type *dmalloc_##type(size_t, nanos6_data_distribution_t, size_t, size_t *);

#define DFREE_FUNC_TYPE_DECL(type) \
void dfree_##type(type *, size_t);



/* Functions for 'double' */
LMALLOC_FUNC_TYPE_DECL(double);
LFREE_FUNC_TYPE_DECL(double);
DMALLOC_FUNC_TYPE_DECL(double);
DFREE_FUNC_TYPE_DECL(double);


/* We 're done with these */
#undef LMALLOC_FUNC_TYPE_DECL
#undef LFREE_FUNC_TYPE_DECL
#undef DMALLOC_FUNC_TYPE_DECL
#undef DFREE_FUNC_TYPE_DECL

#endif /* __OMPSS_CLUSTER_BENCHMARKS_MEMORY_H__ */

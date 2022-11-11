#include "memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Memory allocation functions generators */
#define LMALLOC_FUNC_TYPE(type)	\
type *lmalloc_##type(size_t nr_elements) 			\
{ 								\
	return _lmalloc(nr_elements * sizeof(type));	\
}

#define LFREE_FUNC_TYPE(type) \
void lfree_##type(type *ptr, size_t nr_elements)	\
{							\
	_lfree(ptr, nr_elements * sizeof(type));	\
}

#define DMALLOC_FUNC_TYPE(type) \
type *dmalloc_##type(size_t nr_elements, nanos6_data_distribution_t policy,	\
		size_t nr_dimensions, size_t *dimensions)			\
{										\
	return _dmalloc(nr_elements * sizeof(type), policy, nr_dimensions, 	\
			dimensions);						\
}

#define DFREE_FUNC_TYPE(type) \
void dfree_##type(type *ptr, size_t nr_elements)	\
{							\
	_dfree(ptr, nr_elements * sizeof(type));	\
}


/* Type-agnostic wrappers of Nanos6 memory API */
static void *_lmalloc(size_t size)
{
	void *ret = nanos6_lmalloc(size);
	if (!ret) {
		fprintf(stderr, "Could not allocate Nanos6 local memory\n");
		exit(1);
	}

	return ret;
}

static void _lfree(void *ptr, size_t size)
{
	nanos6_lfree(ptr, size);
}

static void *_dmalloc(size_t size, nanos6_data_distribution_t policy,
		size_t nr_dimensions, size_t *dimensions)
{
	void *ret = nanos6_dmalloc(size, policy, nr_dimensions, dimensions);
	if (!ret) {
		fprintf(stderr,
			"Could not allocate Nanos6 distributed memory\n");
		exit(1);
	}

	return ret;
}

static void _dfree(void *ptr, size_t size)
{
	nanos6_dfree(ptr, size);
}


/* Functions for 'doubles' */
LMALLOC_FUNC_TYPE(double);
LFREE_FUNC_TYPE(double);
DMALLOC_FUNC_TYPE(double);
DFREE_FUNC_TYPE(double);

#undef LMALLOC_FUNC_TYPE
#undef LFREE_FUNC_TYPE
#undef DMALLOC_FUNC_TYPE
#undef DFREE_FUNC_TYPE

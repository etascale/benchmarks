/* AUTORIGHTS
Copyright (C) 2007 Princeton University
      
This file is part of Ferret Toolkit.

Ferret Toolkit is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

OmpSs/OpenMP 4.0 versions written by Dimitrios Chasapis and Iulian Brumar - Barcelona Supercomputing Center
*/
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>
#include <cass.h>
#include <cass_timer.h>
#include <../image/image.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

#define DEFAULT_DEPTH	25
#define MAXR	100
#define IMAGE_DIM	14

#define MAX_QUERIES 	4000

const char *db_dir = NULL;
const char *table_name = NULL;
const char *query_dir = NULL;
const char *output_path = NULL;

FILE *fout;

int DEPTH = DEFAULT_DEPTH;

int top_K = 10;

char *extra_params = "-L 8 - T 20";

cass_env_t *env;
cass_table_t *table;
cass_table_t *query_table;

int vec_dist_id = 0;
int vecset_dist_id = 0;

struct load_data
{
	int width, height;
	char *name;
	unsigned char *HSV, *RGB;
};


struct seg_data
{
	int width, height, nrgn;
	char *name;
	unsigned char *mask;
	unsigned char *HSV;
};


struct extract_data
{
	cass_dataset_t ds;
	char *name;
};


struct vec_query_data
{
	char *name;
	cass_dataset_t *ds;
	cass_result_t result;
};


struct rank_data
{
	char *name;
	cass_dataset_t *ds;
	cass_result_t result;
};



/* ------- The Helper Functions ------- */
int total_queries;
int ten_pers = MAX_QUERIES/10;
int window = MAX_QUERIES/10;

struct load_data *loaded_images[MAX_QUERIES];
struct seg_data *segmented_images[MAX_QUERIES];
struct extract_data *extracted_data[MAX_QUERIES]; 
struct vec_query_data *vectorized_data[MAX_QUERIES]; 
struct rank_data *ranked_results[MAX_QUERIES];

struct load_data *data_load(const char *file);
struct seg_data *t_seg(struct load_data*);
struct extract_data *t_extract(struct seg_data*);
struct vec_query_data *t_vec(struct extract_data*);
struct rank_data *t_rank(struct vec_query_data*);
void t_out (struct rank_data *, FILE *);


char path[BUFSIZ];


int scan_dir (const char *, char *head);

int dir_helper (char *dir, char *head)
{
	DIR *pd = NULL;
	struct dirent *ent = NULL;
	int result = 0;
	pd = opendir(dir);
	if (pd == NULL) goto except;
	for (;;)
	{
		ent = readdir(pd);
		if (ent == NULL) break;
		if (scan_dir(ent->d_name, head) != 0) return -1;
	}
	goto final;

except:
	result = -1;
	perror("Error:");
final:
	if (pd != NULL) closedir(pd);
	return result;
}

int centinel[16];
int read_resource;

/* the whole path to the file */
int file_helper (const char *file)
{
	int r;
	struct load_data *data;



	data = (struct load_data *)malloc(sizeof(struct load_data));
	assert(data != NULL);

	data->name = strdup(file);

//JL: taskify read operation: ensure at most four cncurrent reads
//	r = image_read_rgb_hsv(file, &data->width, &data->height, &data->RGB, &data->HSV);

        read_resource = total_queries%4;
#pragma omp task private(r) firstprivate(data, total_queries) \
                 depend(inout: loaded_images[total_queries], centinel[read_resource]) shared (loaded_images)
{
	r = image_read_rgb_hsv(data->name, &data->width, &data->height, &data->RGB, &data->HSV);
	assert(r == 0);

	/*
	r = image_read_rgb(file, &data->width, &data->height, &data->RGB);
	r = image_read_hsv(file, &data->width, &data->height, &data->HSV);
	*/

	loaded_images[total_queries] = data;
}
//JL: end
	
	#pragma omp task depend(in: loaded_images[total_queries]) depend(out: segmented_images[total_queries]) firstprivate(total_queries) //label(t_seg)
	segmented_images[total_queries] = t_seg(loaded_images[total_queries]);
	#pragma omp task depend(in: segmented_images[total_queries]) depend(out: extracted_data[total_queries]) firstprivate(total_queries) //label(t_extract)
	extracted_data[total_queries] = t_extract(segmented_images[total_queries]);
	#pragma omp task depend(in: extracted_data[total_queries]) depend(out: vectorized_data[total_queries]) firstprivate(total_queries) //label(t_vec)
	vectorized_data[total_queries] = t_vec(extracted_data[total_queries]);
	#pragma omp task depend(in: vectorized_data[total_queries]) depend(out: ranked_results[total_queries]) firstprivate(total_queries) //label(t_rank)
	ranked_results[total_queries] = t_rank(vectorized_data[total_queries]);
	#pragma omp task depend(in: ranked_results[total_queries]) depend(inout: fout) firstprivate(total_queries) //label(t_out)
	t_out(ranked_results[total_queries], fout);
	//only work for a small window of tasks
// 	if(total_queries == window) {
// 	 window += ten_pers;
// 	 #pragma omp taskwait
// 	}
	
	total_queries++;

	return 0;
}

int scan_dir (const char *dir, char *head)
{
	struct stat st;
	int ret;
	/* test for . and .. */
	if (dir[0] == '.')
	{
		if (dir[1] == 0) return 0;
		else if (dir[1] == '.')
		{
			if (dir[2] == 0) return 0;
		}
	}

	/* append the name to the path */
	strcat(head, dir);
	ret = stat(path, &st);
	if (ret != 0)
	{
		perror("Error:");
		return -1;
	}
	if (S_ISREG(st.st_mode)) file_helper(path);
	else if (S_ISDIR(st.st_mode))
	{
		strcat(head, "/");
		dir_helper(path, head + strlen(head));
	}
	/* removed the appended part */
	head[0] = 0;
	return 0;
}


/* ------ The Stages ------ */
void t_load (char *dir)
{
	//const char *dir = (const char *)dummy;
	path[0] = 0;

	if (strcmp(dir, ".") == 0)
	{
		dir_helper(".", path);
	}
	else
	{
		scan_dir(dir, path);
	}

	return;

}

struct seg_data *t_seg (struct load_data *load)
{
	struct seg_data *seg;
	//struct load_data *load;

	assert(load != NULL);
	seg = (struct seg_data *)calloc(1, sizeof(struct seg_data));

	seg->name = load->name;

	seg->width = load->width;
	seg->height = load->height;
	seg->HSV = load->HSV;
	image_segment(&seg->mask, &seg->nrgn, load->RGB, load->width, load->height);

	free(load->RGB);
	free(load);

	return seg;

}

struct extract_data *t_extract (struct seg_data *seg)
{
	//struct seg_data *seg;
	struct extract_data *extract;

	assert(seg != NULL);
	extract = (struct extract_data *)calloc(1, sizeof(struct extract_data));

	extract->name = seg->name;

	image_extract_helper(seg->HSV, seg->mask, seg->width, seg->height, seg->nrgn, &extract->ds);

	free(seg->mask);
	free(seg->HSV);
	free(seg);

	return extract;
}

struct vec_query_data *t_vec (struct extract_data *extract)
{
	//struct extract_data *extract;
	struct vec_query_data *vec;
	cass_query_t query;

	assert(extract != NULL);
	vec = (struct vec_query_data *)calloc(1, sizeof(struct vec_query_data));
	vec->name = extract->name;

	memset(&query, 0, sizeof query);
	query.flags = CASS_RESULT_LISTS | CASS_RESULT_USERMEM;

	vec->ds = query.dataset = &extract->ds;
	query.vecset_id = 0;

	query.vec_dist_id = vec_dist_id;

	query.vecset_dist_id = vecset_dist_id;

	query.topk = 2*top_K;

	query.extra_params = extra_params;

	cass_result_alloc_list(&vec->result, vec->ds->vecset[0].num_regions, query.topk);

//	cass_result_alloc_list(&result_m, 0, T1);
//	cass_table_query(table, &query, &vec->result);
	cass_table_query(table, &query, &vec->result);

	return vec;
}

struct rank_data *t_rank (struct vec_query_data *vec)
{
	//struct vec_query_data *vec;
	struct rank_data *rank;
	cass_result_t *candidate;
	cass_query_t query;

	assert(vec != NULL);

	rank = (struct rank_data *)calloc(1, sizeof(struct rank_data));
	rank->name = vec->name;

	query.flags = CASS_RESULT_LIST | CASS_RESULT_USERMEM | CASS_RESULT_SORT;
	query.dataset = vec->ds;
	query.vecset_id = 0;

	query.vec_dist_id = vec_dist_id;

	query.vecset_dist_id = vecset_dist_id;

	query.topk = top_K;

	query.extra_params = NULL;

	candidate = cass_result_merge_lists(&vec->result, (cass_dataset_t *)query_table->__private, 0);
	query.candidate = candidate;


	cass_result_alloc_list(&rank->result, 0, top_K);
	cass_table_query(query_table, &query, &rank->result);

	cass_result_free(&vec->result);
	cass_result_free(candidate);
	free(candidate);
	cass_dataset_release(vec->ds);
	free(vec->ds);
	free(vec);

	return rank;
}
void t_out (struct rank_data *rank, FILE *stream)
{
	//struct rank_data *rank;

	assert(rank != NULL);


	fprintf(stream, "%s", rank->name);

	ARRAY_BEGIN_FOREACH(rank->result.u.list, cass_list_entry_t p)
	{
		char *obj = NULL;
		if (p.dist == HUGE) continue;
		cass_map_id_to_dataobj(query_table->map, p.id, &obj);
		assert(obj != NULL);
		fprintf(stream, "\t%s:%g", obj, p.dist);
	} ARRAY_END_FOREACH;

	fprintf(stream, "\n");

	cass_result_free(&rank->result);
	free(rank->name);
	free(rank);

	/*
	if (input_end && (cnt_enqueue == cnt_dequeue))
	{
		pthread_mutex_lock(&done_mutex);
		output_end = 1;
		pthread_cond_signal(&done);
		pthread_mutex_unlock(&done_mutex);
	}
	*/
	
	return;
}

int main (int argc, char *argv[])
{
	stimer_t tmr;

	int ret, i;

        printf("PARSEC Benchmark Suite\n");  
	 	fflush(NULL);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_ferret);
#endif

	if (argc < 8)
	{
		printf("%s <database> <table> <query dir> <top K> <depth> <n> <out>\n", argv[0]);
		printf("Warning: Argument n is ignored!\n");
		return 0;
	}

	db_dir = argv[1];
	table_name = argv[2];
	query_dir = argv[3];
	top_K = atoi(argv[4]);

	DEPTH = atoi(argv[5]);
	
	output_path = argv[7];

	fout = fopen(output_path, "w");
	assert(fout != NULL);

	cass_init();

//	printf("This is the sizeof rank %i\n ", sizeof(struct rank_data));

	ret = cass_env_open(&env, db_dir, 0);
	if (ret != 0) { printf("ERROR: %s\n", cass_strerror(ret)); return 0; }

	vec_dist_id = cass_reg_lookup(&env->vec_dist, "L2_float");
	assert(vec_dist_id >= 0);

	vecset_dist_id = cass_reg_lookup(&env->vecset_dist, "emd");
	assert(vecset_dist_id >= 0);

	i = cass_reg_lookup(&env->table, table_name);


	table = query_table = cass_reg_get(&env->table, i);

	i = table->parent_id;

	if (i >= 0)
	{
		query_table = cass_reg_get(&env->table, i);
	}

	if (query_table != table) cass_table_load(query_table);
	
	cass_map_load(query_table->map);

	cass_table_load(table);

	image_init(argv[0]);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif

	stimer_tick(&tmr);
	int startt = time(NULL);
	
	/* 
	   Create shared arrays here  
	   Each array replaces the queue of the last pipeline stage
	   It is not exactly a pipeline.  We do not have a dedicated
	   thread pool at each stage, instead, if the runtime uses
	   a thread pool, we have tasks, that will be served when the
	   corresponding task of the previous state has finished, thus
	   its input is available.
	 */
	
	int max_queries = MAX_QUERIES;
	t_load(query_dir);
/*
	struct seg_data **segmented_images = 
	  (struct seg_data**)malloc(sizeof(struct seg_data*)*total_queries);
	struct extract_data **extracted_data = 
	  (struct extract_data**)malloc(sizeof(struct extract_data*)*total_queries);
	struct vec_query_data **vectorized_data = 
	  (struct vec_query_data**)malloc(sizeof(struct vec_query_data*)*total_queries);
	struct rank_data **ranked_results = 
	  (struct rank_data**)malloc(sizeof(struct rank_data*)*total_queries);

	for(i=0; i < total_queries; i++) {
	  #pragma omp task in(loaded_images[i]) out(segmented_images[i]) firstprivate(i) label(t_seg)
	  segmented_images[i] = t_seg(loaded_images[i]);
	}
	for(i=0; i < total_queries; i++) {
	  #pragma omp task in(segmented_images[i]) out(extracted_data[i]) firstprivate(i) label(t_extract)
	  extracted_data[i] = t_extract(segmented_images[i]);
	}
	for(i=0; i < total_queries; i++) {
	  #pragma omp task in(extracted_data[i]) out(vectorized_data[i]) firstprivate(i) label(t_vec)
	  vectorized_data[i] = t_vec(extracted_data[i]);
	}
	for(i=0; i < total_queries; i++) {
	  #pragma omp task in(vectorized_data[i]) out(ranked_results[i]) firstprivate(i) label(t_rank)
	  ranked_results[i] = t_rank(vectorized_data[i]);
	}
	#pragma omp taskwait
*/
	#pragma omp taskwait
// 	for(i=0; i < total_queries; i++) {
// 	  t_out(ranked_results[i]);
// 	}
	
	stimer_tuck(&tmr, "QUERY TIME");

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif
	
	ret = cass_env_close(env, 0);
	if (ret != 0) { printf("ERROR: %s\n", cass_strerror(ret)); return 0; }
	
	cass_cleanup();

	image_cleanup();

	fclose(fout);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

	return 0;
}



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
*/
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <cass.h>
#include <cass_stat.h>
#include <cass_timer.h>
#include <../image/image.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

#define MAXR	100
#define IMAGE_DIM	14

const char *db_dir = NULL;
const char *table_name = NULL;
const char *query_dir = NULL;
const char *output_path = NULL;

FILE *fout;

int top_K = 10;

char *extra_params = "-L 8 - T 20";

int input_end;
pthread_cond_t done;
pthread_mutex_t done_mutex;

cass_env_t *env;
cass_table_t *table;
cass_table_t *query_table;

int vec_dist_id = 0;
int vecset_dist_id = 0;

int NTHREAD = 1;
int DEPTH = 1;

/*Because results are written at the end, a rank_data struct array will be used
In practice all the used fields are name and result. That's why in out stage only those are freed
*/
/*
struct rank_data
{
        char *name;
        cass_dataset_t *ds;
        cass_result_t result;
//        struct rank_data *QUEUE_LINK;
};*/


struct rank_data
{
        char *name;
        cass_dataset_t *ds;
        cass_result_t result;
        struct rank_data *QUEUE_LINK;
};


struct rank_data rank_vector[3500];
int num_image = 0;

/* ------- The Helper Functions ------- */
char path[BUFSIZ];


void do_query (const char *);
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
	if (S_ISREG(st.st_mode)) {
		do_query(path);
	}
	else if (S_ISDIR(st.st_mode))
	{
		strcat(head, "/");
		dir_helper(path, head + strlen(head));
	}
	/* removed the appended part */
	head[0] = 0;
	return 0;
}

void *t_out ()
{
	int i;
        for (i = 0; i < num_image; ++i)
        {

                fprintf(fout, "%s", (rank_vector[i]).name);
        //      printf("t_out %i\n", (int)pthread_self());

                ARRAY_BEGIN_FOREACH(rank_vector[i].result.u.list, cass_list_entry_t p)
                {
                        char *obj = NULL;
                        if (p.dist == HUGE) continue;
                        cass_map_id_to_dataobj(query_table->map, p.id, &obj);
                        assert(obj != NULL);
                        fprintf(fout, "\t%s:%g", obj, p.dist);
                } ARRAY_END_FOREACH;

                fprintf(fout, "\n");

                cass_result_free(&(rank_vector[i].result)); //verificar que aix?? estigui be
                free(rank_vector[i].name);
                //free(&(rank_vector[i]));
		//QUAN S'ALLIBERA LA RESTA d'ELEMENTS DEL STRUCT
        }
        return NULL;
}


/* ------ The Stages ------ */
void scan (void)
{
	const char *dir = query_dir;

	path[0] = 0;

	if (strcmp(dir, ".") == 0)
	{
		dir_helper(".", path);
	}
	else
	{
		scan_dir(dir, path);
	}
	//printf("Soc AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA%i\n", omp_get_thread_num());
	#pragma omp taskwait
	t_out();
}

//#pragma omp task
void do_query (const char *name)
{
	cass_dataset_t ds;
	cass_query_t query;
	cass_result_t result;
	cass_result_t *candidate;
	//printf("Soc AAAAAAAAAAAAAAAAAAAAAAAAAAAAAa %i\n", omp_get_thread_num());

	//{

	unsigned char *HSV, *RGB;
	unsigned char *mask;
	int width, height, nrgn;
	int r;
	int nameSize = strlen(name) +1;
	char *auxName = malloc(nameSize*sizeof(char));
	strcpy(auxName, name);
	r = image_read_rgb_hsv(name, &width, &height, &RGB, &HSV);
	rank_vector[num_image].name = auxName;
	assert(r == 0);
	printf("I've tried to print name %s and its size %i\n", name, nameSize);
	#pragma omp task firstprivate(width, height, nrgn, mask, candidate, HSV, RGB, ds, query, result, num_image)
	{
//in(HSV[ : ], RGB[ : ], ds, query, result)  
	image_segment(&mask, &nrgn, RGB, width, height);

	image_extract_helper(HSV, mask, width, height, nrgn, &ds);
	printf("problem1\n");
	/* free image & map */
	free(HSV);
	free(RGB);
	free(mask);
	printf("problem2\n");
	
	//}

	memset(&query, 0, sizeof query);
	printf("problem3\n");
	query.flags = CASS_RESULT_LISTS | CASS_RESULT_USERMEM;

	query.dataset = &ds;
	query.vecset_id = 0;

	query.vec_dist_id = vec_dist_id;

	query.vecset_dist_id = vecset_dist_id;

	query.topk = 2*top_K;

	query.extra_params = extra_params;

	cass_result_alloc_list(&result, ds.vecset[0].num_regions, query.topk);

	cass_table_query(table, &query, &result);

	memset(&query, 0, sizeof query);
	printf("problem4\n");

	query.flags = CASS_RESULT_LIST | CASS_RESULT_USERMEM | CASS_RESULT_SORT;
	query.dataset = &ds;
	query.vecset_id = 0;

	query.vec_dist_id = vec_dist_id;

	query.vecset_dist_id = vecset_dist_id;

	query.topk = top_K;

	query.extra_params = NULL;

	candidate = cass_result_merge_lists(&result, (cass_dataset_t *)query_table->__private, 0);
	printf("problem5\n");
	query.candidate = candidate;

	cass_result_free(&result);
	printf("problem6\n");

	cass_result_alloc_list(&(rank_vector[num_image].result), 0, top_K);
	printf("problem7\n");
	//el problema esta aqui
	cass_table_query(query_table, &query, &(rank_vector[num_image].result));//aqui has obtingut el result. El nom ja esta assignat
	printf("problem8\n");

	cass_result_free(candidate);
	printf("problem9\n");
	free(candidate);
	printf("problem10\n");
	cass_dataset_release(&ds);
	printf("problem11\n");

/*

	fprintf(fout, "%s", auxName);
	//printf("I've tried to print name %s\n", auxName);
	ARRAY_BEGIN_FOREACH(result.u.list, cass_list_entry_t p)
	{
		char *obj = NULL;
		if (p.dist == HUGE) continue;
		cass_map_id_to_dataobj(query_table->map, p.id, &obj);
		assert(obj != NULL);
		fprintf(fout, "\t%s:%g", obj, p.dist);
	} ARRAY_END_FOREACH;

	fprintf(fout, "\n");

	cass_result_free(&result);*/
	}
	#pragma omp taskwait
	++num_image;
}

int main (int argc, char *argv[])
{
	stimer_t tmr;
	int ret, i;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n");
        fflush(NULL);
#else
        printf("PARSEC Benchmark Suite\n");
        fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_ferret);
#endif

	if (argc < 8)
	{
		printf("%s <database> <table> <query dir> <top K> <depth> <n> <out>\n", argv[0]); 
		return 0;
	}

	db_dir = argv[1];
	table_name = argv[2];
	query_dir = argv[3];
	top_K = atoi(argv[4]);

	DEPTH = atoi(argv[5]);
	NTHREAD = atoi(argv[6]);
	if(NTHREAD != 1) {
		printf("n must be 1 (serial version)\n");
		exit(1);
	}

	output_path = argv[7];

	fout = fopen(output_path, "w");
	assert(fout != NULL);

	//rank_vector = (rank_data *)malloc(sizeof(rank_data)*top_K);	

	cass_init();

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

	stimer_tick(&tmr);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif
	scan();
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

	stimer_tuck(&tmr, "QUERY TIME");

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


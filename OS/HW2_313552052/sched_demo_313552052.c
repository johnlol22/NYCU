#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
typedef struct {
	int thread_id;
	double busy_time;
	char *policy;
	char *priority;
} thread_attr;

pthread_barrier_t barrier;

void *thread_func(void *arg)
{
	
	thread_attr p = *(thread_attr*)arg;
	int rc = pthread_barrier_wait(&barrier);

	double wait = p.busy_time;
	int id = p.thread_id;
	/*
	cpu_set_t check;
	pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &check);
	for (int i=0;i<CPU_SETSIZE;i++)
	{
		if (CPU_ISSET(i, &check))
				printf("thread %d run on %d\n", id, i);
	}
	{
		int policy = sched_getscheduler(0);
		printf("%d's policy %d\n", id, policy);
		struct sched_param cur_param;
		sched_getparam(0, &cur_param);
		printf("%d's priority %d\n", id, cur_param.sched_priority);
	}
	*/
	for (int i=0;i<3;i++)
	{
		printf("Thread %d is starting\n", id);
		clock_t start_time = clock();
		while (clock() - start_time < wait*CLOCKS_PER_SEC);
	}
	return;
}
int main(int argc, char *argv)
{
	int opt;
	int num_thread;
	double t;
	char *policy;
	char *priority;
	while((opt = getopt(argc, argv, "n:t:s:p:")) != -1)
	{
		switch (opt)
		{
		case 'n':
			num_thread = atoi(optarg);
			break;
		case 't':
			t = atof(optarg);
			break;
		case 's':
			policy = optarg;
			break;
		case 'p':
			priority = optarg;
			break;
		}
	}
	pthread_t th[num_thread];
	pthread_attr_t attr;
	struct sched_param param;

	thread_attr pa[num_thread];

	char *thread_policy = strtok(policy, ",");
	for (int i=0;i<num_thread;i++)
	{
		pa[i].thread_id = i;
		pa[i].policy = thread_policy;
		pa[i].busy_time = t;
		thread_policy = strtok(NULL, ",");
	}
	char *thread_priority = strtok(priority, ",");
	for (int i=0;i<num_thread;i++)
	{
		pa[i].priority = thread_priority;
		thread_priority = strtok(NULL, ",");
	}
	
	
	pthread_barrier_init(&barrier, NULL, num_thread);
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(0, &cpuset);
	sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
	char *fifo = "FIFO";
	for (int i=0;i<num_thread;i++)
	{
		pthread_attr_init(&attr);
		if (strcmp(pa[i].policy, fifo) == 0)
			pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
		else
			pthread_attr_setschedpolicy(&attr, SCHED_OTHER);

		param.sched_priority = atoi(pa[i].priority);
		pthread_attr_setschedparam(&attr, &param);
		pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

		int rc = pthread_create(&th[i], &attr, thread_func, &pa[i]);
		if (rc != 0)
		{
			printf("pthread create failed...\n");
			fprintf(stderr, "Error creating thread %s\n", strerror(rc));
			return 1;
		}
	}
	void *res;
	for (int i=0;i<num_thread;i++)
	{
		pthread_join(th[i], &res);
	}
	pthread_barrier_destroy(&barrier);
	pthread_attr_destroy(&attr);
}

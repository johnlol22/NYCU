#include <iostream>
#include <time.h>
#include <cstdlib>
#include <pthread.h>

pthread_mutex_t mutex;
long long int ans = 0;
void *runner(void *arg)
{
	long long int sum = 0;
	double x, y;
	long long int *step =(long long int *)arg;
	unsigned int local_seed = time(NULL);
	for (long long int i=0;i<*step;i++){
		x = (double)2*rand_r(&local_seed) / (RAND_MAX + 1.0) - 1.0;
		y = (double)2*rand_r(&local_seed) / (RAND_MAX + 1.0) - 1.0;
		if (x*x+y*y <= 1.0){
			sum++;
		}
	}

	pthread_mutex_lock(&mutex);
	ans += sum;
	pthread_mutex_unlock(&mutex);

	pthread_exit(NULL);
}
int main(int argc, char *argv[])
{
	if (argc != 3){
		std::cout<<"Error parameters !"<<std::endl;
		return -1;
	}
	long long int toss = atoi(argv[2]);
	int t_num = atoi(argv[1]);
	long long int step = toss / t_num;	
	int remaining = toss % t_num;
	pthread_t t[t_num];
	srand(time(NULL));
	for (int i=0;i<t_num;i++){
		if (pthread_create(&t[i], NULL, runner, &step) != 0){
			perror("creation fail.");
		}
	}
	for (int i=0;i<t_num;i++){
		pthread_join(t[i], NULL);
	}
	while (remaining > 0){
		double x = 2 * rand() / (RAND_MAX + 1.0) - 1.0;
		double y = 2 * rand() / (RAND_MAX + 1.0) - 1.0;
		if (x*x+y*y <= 1){
			ans++;
		}
		remaining--;
	}
	pthread_mutex_destroy(&mutex);
	std::cout<<4*ans/(double)toss<<std::endl;
}

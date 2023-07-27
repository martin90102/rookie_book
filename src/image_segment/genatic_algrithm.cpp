/**
 * @file genatic_algrithm.cpp
 * @author luan yeming
 * @brief  新手教材(应用)
 * @version 0.1
 * @date 2023-07-25
 * @note https://github.com/martin90102/rookie_book
 * @copyright Copyright (c) 2023
 *
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

Mat img;
int popsize = 10;
int lchrom = 8;
double cross_rate = 0.7;
double mutation_rate = 0.4;
int maxgen = 150;
Mat oldpop;
Mat oldfitness;

void initpop(Mat& pop);
void fitness_order(Mat& pop, Mat& fitness);
void select(Mat& pop, Mat& fitness);
void crossover(Mat& pop);
void mutation(Mat& pop);
void generation(Mat& pop);

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "give correct arguments in format below!" << endl;
        cout << "Usage: " << argv[0] << " <image>" << endl;
        return -1;
    }
    string path = argv[1];
    img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat pop = Mat::zeros(popsize, lchrom, CV_32F);
    initpop(pop);

    for (int gen = 0; gen < maxgen; gen++)
    {
        generation(pop);
    }

    Mat best = Mat::zeros(1, lchrom, CV_32F);
    Mat bestfitness = Mat::zeros(1, 1, CV_32F);
    fitness_order(pop, bestfitness);
    best = pop.row(0).clone();
    int thr = 0;
    for (int j = 0; j < lchrom; j++)
    {
        thr += best.at<float>(0, j) * pow(2, lchrom - j - 1);
    }
    thr = thr * 255 / (pow(2, lchrom) - 1);
    Mat binary;
    threshold(img, binary, thr, 255, THRESH_BINARY);
    imshow("output", binary);
    imshow("gray", img);
    waitKey(0);
}
/**
 * @brief 初始化种群
 * 
 * @param pop 
 */
void initpop(Mat& pop)
{
    for (int i = 0; i < popsize; i++)
    {
        for (int j = 0; j < lchrom; j++)
        {
            pop.at<float>(i, j) = rand() % 2;
        }
    }
}
/**
 * @brief 计算种群中每个个体的适应度，并对种群进行排序
 * 
 * @param pop 
 * @param fitness 
 */
void fitness_order(Mat& pop, Mat& fitness)
{
    for (int i = 0; i < popsize; i++)
    {
        int thr = 0;
        for (int j = 0; j < lchrom; j++)
        {
            thr += pop.at<float>(i, j) * pow(2, lchrom - j - 1);
        }
        thr = thr * 255 / (pow(2, lchrom) - 1);

        Mat binary;
        threshold(img, binary, thr, 255, THRESH_BINARY);
        Scalar mean1, mean2, stddev1, stddev2;
        meanStdDev(img, mean1, stddev1, binary == 0);
        meanStdDev(img, mean2, stddev2, binary == 255);
        fitness.at<float>(0, i) = abs(mean1[0] - mean2[0]);
    }

    for (int i = 0; i < popsize - 1; i++)
    {
        for (int j = i + 1; j < popsize; j++)
        {
            if (fitness.at<float>(0, i) < fitness.at<float>(0, j))
            {
                Mat temp = pop.row(i).clone();
                pop.row(i) = pop.row(j).clone();
                pop.row(j) = temp.clone();

                float tempFitness = fitness.at<float>(0, i);
                fitness.at<float>(0, i) = fitness.at<float>(0, j);
                fitness.at<float>(0, j) = tempFitness;
            }
        }
    }
}

/**
 * @brief 选择操作，将新种群中的个体与旧种群中的个体进行比较，选择适应度高的个体
 * 
 * @param pop 
 * @param fitness 
 */
void select(Mat& pop, Mat& fitness)
{
    oldpop = pop.clone();
    oldfitness = fitness.clone();

    for (int i = 0; i < popsize; i++)
    {
        if (fitness.at<float>(0, i) < oldfitness.at<float>(0, i))
        {
            pop.row(i) = oldpop.row(i).clone();
            fitness.at<float>(0, i) = oldfitness.at<float>(0, i);
        }
    }
}
/**
 * @brief 交叉操作，随机选择两个个体，以一定的概率进行交叉
 * 
 * @param pop 
 */
void crossover(Mat& pop)
{
    for (int i = 0; i < popsize; i += 2)
    {
        if (i + 1 < popsize) 
        {
            float p = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if (p < cross_rate)
            {
                int cutpoint = rand() % lchrom;
                Mat temp1 = pop.row(i).clone();
                Mat temp2 = pop.row(i + 1).clone();
                temp1.colRange(cutpoint, lchrom).copyTo(pop.row(i).colRange(cutpoint, lchrom));
                temp2.colRange(cutpoint, lchrom).copyTo(pop.row(i + 1).colRange(cutpoint, lchrom));
            }
        }
    }
}
/**
 * @brief 变异操作，随机选择一个个体，以一定的概率进行变异
 * 
 * @param pop 
 */
void mutation(Mat& pop)
{
    int total_genes = popsize * lchrom;
    int num_mutations = round(mutation_rate * total_genes);
    for (int i = 0; i < num_mutations; i++)
    {
        int rand_individual = rand() % popsize;
        int rand_gene = rand() % lchrom;
        pop.at<float>(rand_individual, rand_gene) = 1 - pop.at<float>(rand_individual, rand_gene);
    }
}
/**
 * @brief 进化
 * 
 * @param pop 
 */
void generation(Mat& pop)
{
    Mat fitness = Mat::zeros(1, popsize, CV_32F);
    fitness_order(pop, fitness);
    select(pop, fitness);
    crossover(pop);
    mutation(pop);
}
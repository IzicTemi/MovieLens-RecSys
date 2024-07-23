# Matrix Factorization for Recommender Systems: A MovieLens Case Study

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Future Work](#future-work)
7. [References](#references)

## Overview

This study investigates the use of matrix factorization in building a movie recommendation system using the MovieLens 25M dataset. Matrix factorization techniques are employed to decompose the user-item interaction matrix into latent factors, which help predict user preferences and recommend movies accurately. The system integrates user and item biases to improve the recommendation quality, addressing the inherent sparsity and variability in user ratings. Regularization parameters are fine-tuned to prevent overfitting and enhance the model's generalization capabilities. Evaluation results demonstrate the effectiveness of the approach, showing the ability of the model to give personalized recommendations from ratings only without any contextual understanding.

### Features
- Matrix factorization for collaborative filtering
- User and item bias integration
- Feature embedding to address cold-start problems
- Alternating Least Squares (ALS) optimization
- Visualization of learned embeddings

## Dataset

The MovieLens 25M dataset contains:
- 25,000,097 ratings
- 62,423 movies
- 162,541 users
- Ratings collected between January 09, 1995 and November 21, 2019

## Exploratory Data Analysis

### Rating Distribution
![Rating Distribution](https://github.com/user-attachments/assets/f53a99ad-d17b-4cf9-8a1d-ce35a84165b3)

The ratings are distributed over a predefined scale from 0.5 to 5 stars. Higher ratings are more prevalent, possibly due to users' tendency to watch movies they expect to enjoy.

### Top 10 Most Watched Movies
![Top 10 Movies](https://github.com/user-attachments/assets/190d3d3b-3c02-4923-a68e-c41eeafce809)


This chart provides insight into user preferences and movie popularity.

### Power Laws
![Power Laws](https://github.com/user-attachments/assets/d783a28e-b445-4e37-98b4-e462b321d099)


The frequency of user interactions and item popularity follows a power law distribution, indicating that a small number of items are rated extremely frequently, while the majority have relatively few ratings.

## Methodology

### Data Structuring
We implemented a double-indexed data structure for efficient access and indexing of user-item interactions.

### Matrix Factorization Model
The core of our recommendation system is a matrix factorization model that decomposes the user-item interaction matrix into lower-dimensional user and item latent features.

#### Objective Function
We use a regularized negative log-likelihood loss function:

$$
L = - \frac{\lambda}{2} \sum_{m=1}^M \sum_{n \in \Pi(n)} \left(r_{mn} - \left(u_m^T v_n + b_m^{(u)} + b_n^{(i)}\right)\right)^2 - \frac{\tau}{2} \sum_m u_m^T u_m - \frac{\tau}{2} \sum_n v_n^T v_n - \frac{\gamma}{2} \sum_m \left(b_m^{(u)}\right)^2 - \frac{\gamma}{2} \sum_n \left(b_n^{(i)}\right)^2
$$

Where:
- $r_{mn}$ is the rating of user $m$ for item $n$
- $u_m$ and $v_n$ are the latent factors for user $m$ and item $n$ respectively
- $b_m^{(u)}$ and $b_n^{(i)}$ are the user and item biases
- $\lambda$, $\tau$, and $\gamma$ are regularization parameters

#### Alternating Least Squares (ALS)
We utilize the ALS approach to minimize the loss function, iteratively fixing one set of variables and solving for the other.

### Feature Embedding
To address the cold-start problem, we incorporate movie genre information as feature embeddings.

## Results

### Model Performance

| Model                            | Train RMSE | Test RMSE |
|----------------------------------|------------|-----------|
| Biases Only                      | 0.849      | 0.860     |
| Biases + Vectors (K=2)           | 0.792      | 0.824     |
| Biases + Vectors (K=10)          | 0.706      | 0.806     |
| Biases + Vectors + Features (K=10)| 0.695      | 0.806     |

The matrix factorization model with biases and latent vectors (K=10) achieved the best performance.

### Visualizations

#### Item Embedding (Children vs. Horror Movies)
![Item Embedding](https://github.com/user-attachments/assets/1fa285fa-b779-40da-850e-f6ae51d7acce)


This visualization demonstrates the model's ability to separate different movie genres in the latent space.

#### Feature Embedding
![Feature Embedding](https://github.com/user-attachments/assets/c63ea71c-ad31-46e2-ad1c-4f89c72a2df1)

This plot shows the relationships between different movie genres based on their learned feature embeddings.

### Recommendations
We tested the model by creating a dummy user who rated "Lord of the Rings" five stars. Here are the top recommended movies:

1. Lord of the Rings (1978)
2. The Lord of the Rings: The Return of the King (2003)
3. The Lord of the Rings: The Two Towers (2002)
4. The Hobbit: The Battle of the Five Armies (2014)
5. The Lord of the Rings: The Fellowship of the Ring (2001)

## Future Work

- Explore hybrid models combining matrix factorization with content-based filtering
- Incorporate implicit feedback for more comprehensive user preference modeling
- Investigate advanced techniques to further mitigate the cold-start problem

## References

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), 1-19.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30-37.
- Su, X., & Khoshgoftaar, T. M. (2009). A Survey of Collaborative Filtering Techniques. Advances in Artificial Intelligence, 2009, 1-19.

---
Created by Isaac Temiloluwa OMOLAYO | [Detailed Report](https://drive.google.com/file/d/1wUrgyoqPlTjzdjTSE0rfmwI9wAzNufFb/view?usp=drive_link)

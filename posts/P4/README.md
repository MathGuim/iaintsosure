# Lossy Modeling, Iterative Science and Anomaly Detection [2/2]

## Golems, Fairies, Scree Plots and other mythical creatures

"A pharmaceutical company statistician (working in toxicology) once remarked, 'All this discussion of deleting the outliers is completely backwards. In my work, I usually throw away all the good data, and just analyze the outliers.'" - The American Statistician: Vol. 61, No. 3.

In the previous post, I introduced the problems of Outlier and Anomaly Detection, and distinguished between outliers and anomalies. I showed why you should consider adding more steps to your EDA than just simple charts and statistics, among other things. Today's post's purpose is threefold:

1. Show the benefits of adhering to standard APIs like scikit-learn's when designing custom estimators for your ML experiments.
2. Walk you through my thought process back when I first encountered the problem of Anomaly Detection at work.
3. Show you the difference a simple Feature Engineering trick can make and prove to Data Scientists the need for going further than simple .fit/.predict.

## Time Series Technicalities

Before I start, let me make a conceptual distinction. We can classify outliers as contextual or collective in time-series analysis. An outlier is said to be contextual when its value differs significantly from the values of its temporally adjacent points. It's called collective when a large subset or an entire time series has an unusual shape. This extra complexity is a general phenomenon in all statistical analysis:

"The more complex the data is, the more the analyst has to make prior inferences of what is considered normal for modeling purposes."

## My First Data Science Job

In my first job as a Data Scientist, I was tasked with identifying contextual anomalies in sensor data from the Telecom Industry. We had no ground truth, so the classical approach is to use forecasting. A point is marked as an outlier if it lies outside of its forecasted prediction interval. But my department had had bad results with it before my arrival, and people were adamant against it. So I decided to take some steps back and do some EDA on my own. I chose the most important signal according to my intuition (we had no access to Subject Matter Experts) and plotted it, but what I saw was a blurry picture like this:

Modern plotting libraries like Plotly and interactive web frameworks like Shiny allow you to quickly explore the data interactively. However, there are still limits to the amount of data you can meaningfully plot on the browser. So I used one of the oldest big data tricks in the book: random sampling!

I found something interesting after dozens of random plots. One sample had a flat segment, as shown below. It got me interested in whether that could be a collective anomaly, how common it was, and what had caused it. So I started the systematic literature review phase, seeking inspiration to pivot into this new line of research.

I barely use R nowadays, but I still have the habit of searching for all packages that cover a certain topic, not necessarily to use them, but to have an idea of what kinds of research have been done before. Eventually, I stumbled upon the TSRepr R package, which provides methods for representing time series. So I researched a bit more and found that time-series databases use this kind of trick internally to index and perform similarity search on large volumes of temporal data, so it seemed worth giving it a shot.

I started using the DFT as a representation for my time series. If you don't know what the DFT is, think of it as a way to disintangle the frequencies present in a complex signal by decomposing it as a combination of simpler sinusoidal waves. The efficient algorithm for calculating the DFT is called the FFT, and it is so popular that people use the terms interchangeably. I considered explaining the technical details and intuition behind DFT, but there are already so many good resources out there. You should check out the 3blue1brown series of videos on the topic.

The DFT time-series representation works like this: we use the FFT to extract the frequencies that comprise the original signal. Then we use the inverse FFT to reconstruct the original time series, but using only the first K lower frequencies. By keeping only to the real part, the final result is a smaller signal that retains some of the frequency properties of the original time series. The number K of FFT coefficients determines the amount of compression we get.

We could have used the magnitudes of the K largest FFT coefficients as a representation, but, as we shall see later, the results were quite good with the first method discussed. After reducing the problem's dimensionality, we can use any available Outlier Detection method.

Finding interesting, hidden representations of data is where Deep Learning excels, someone might say. However, I did not know TensorFlow (the main Deep Learning framework back then).

Nevertheless, the DFT representation is also appealing for a couple of reasons: it extends easily to multivariate signals, the FFT algorithm is extremely efficient, and representing signals in terms of their lower frequencies makes intuitive sense.

Statistical methods for Outlier Detection in time series have evolved since then. I haven't kept up with this literature, and this is not the point of the post.

## The Simulation Study

I made the simulation as close as possible to the real data, as I remember it. I simulated 200 signals synthesized from sinusoidal waves with different amplitudes and frequencies and added a bit of random Gaussian noise to make it more realistic. Random flat segments generated with lengths proportional to a Beta Distribution were added in 10% of the signals to simulate the anomalous behavior. All code is available on GitHub.

## The Results

In the experiment, I compare 3 methods for Outlier Detection: PCA, PCA with Mahalanobis distance instead of Euclidean distance and the Local Outlier Factor. I enhance all this methods with the DFT representation. I ran the experiment over 500 random realizations of the Gaussian noise. We can see that by adding the FFT representation as form of Feature Engineering, we consistently increases the performance of all method as measured by the f1_score.

Focusing particurlarly on LOF and picking a specific random realization, we can visualize its Anomaly Scores and the signals with the highest anomaly score. As you can see, the method finds all anoumalous signals in the sample.

## Word of Caution About PCA and Unsupervised Learning in General

In the previous post, I talked about the "compressibility test" for Outlier Detection. The test is possible because of a lesser-known feature of PCA. Once we have the Principal Components, we can reverse-transform them to recreate an "error corrected" version of the data. So PCA can be used for both Error Correction and Outlier Detection.

The scree plot is a common visualization used together with PCA to pick the number of Principal Components. It shows the percentage of "explained variance" by each Principal Component. The Statistical Folklore is to pick the number of Principal Components by looking for a significant "bend" in the scree plot. The problem, however, is that I have yet to see a non-toy example in which the scree plot allowed a clear-cut decision. As you can see in the example below.

A better way to pick the number of components and to validate PCA in general is to use cross-validation: we mask parts of the input and see how well PCA reconstructs them. Somewhat reminiscent of how people train some Large Language Models today. However, even if PCA can reconstruct the original input well, it doesn't mean it is useful for whatever practical application we're planning to use it for, e.g., Anomaly Detection.

Here, it's useful to take a detour showing, in a simplified manner, how PCA works. PCA looks for the projection that maximizes the variance of the transformed data, the so-called Principal Components. Maximizing the variance of the transformed points is equivalent to minimizing the mean-squared reconstruction error due to the Pythagorean Theorem. See this if you need visuals. Think of it like this: let's say that you're trying to buy a car through the internet by looking at photos alone. A photo is a 2D projection of a 3D object. It makes logical sense to look for an angle in the photos where the car is as "spread out" as possible. So PCA assumes the meaningful signal is the one that varies the most; whether this is a good bet or not is usually not something we can verify with data alone. 

Somebody could make a similar argument about LOF, the other unsupervised technique I have used. LOF uses the Euclidean distance to measure locality by default, which makes sense when the signals are in phase and are of the same size. The most meaningful distance metric for your particular case is not something the data alone will tell you, especially not in an unsupervised fashion.

Shadow or Canary Deployment is the most reliable way to evaluate the effectiveness of unsupervised learning. Put it into production for a small subset of the user base and measure its performance against a control group.

## Some Benefits of Adhering to Public APIs Like Scikit-learn's

### Mixins

In Python, a mixin is a class designed to be inherited by another class, but not instantiated on its own. They are used in scikit-learn primarily to avoid code duplication by implementing standardized methods that are shared across different types of objects.

In this experiment, I have used the OutlierMixin for our custom outlier detectors and the TransformerMixin for the FFT representation. The OutlierMixin sets estimator type to "outlier_detector" through its tags (important for data validation and unit tests) and implements the fit_predict method. The ClassNamePrefixFeaturesOutMixin makes transformers compatible with the set_output API (DataFrames instead of numpy arrays). 

All classes also inherit from BaseEstimator, which is the base class for all objects in scikit-learn. It provides the essential get_params() and set_params() methods. Notice that you should inherit first from the Mixins and then from BaseEstimator.

### Unit Tests

Richard McElreath has an awesome talk, "Science as Amateur Software Development," where he argues that researchers have a lot to learn from Software Engineers. If you adhere to the Scikit-Learn API, you get a suite of unit tests for free, as you can see on the experiment repository. You should add a lot more tests, but sklearn.utils.estimator_checks.parametrize_with_checks is a good start. Just a heads-up: Richard uses the term "unit test" in a much broader sense than is commonly used in Software Engineering.

### Open Source Ecosystem

Another benefit to adhering to public APIs is that your code will play nicely with the rest of the Open-Source ecosystem. You can easily use utilities such as sklearn.model_selection.cross_validate. Scikit-Learn's codebase is extensively tested and used by many people around the world, so the chance of bugs is much lower than with a homegrown implementation.

### Numerical Analysis is Hard

Estimation is not easy, and even for things as simple as standardization, there are "tricks" that might make it a little more numerically stable. You shouldn't roll out your own MCMC sampler or automatic differentiation framework, unless you're studying the topic or an expert doing something really unconventional. See this awesome blog post covering tricks like the Welford's Algorithm: "The StandardScaler Isn't Standard".

## Conclusion

In this experiment, we transformed mediocre detectors into SOTA ones for this particular simulation study. I suspect the DFT trick was so effective here not only because we reduced the data dimensionality and ameliorated the Curse of Dimensionality, but also because the anomalies are more unusual in the frequency domain than in the time domain.

As a downside, our uninterpretable, causal-blind "robot" gives us little insight into the reason why a signal was marked as unusual. However, this is not a big deal in this particular application because we're not going to use the models to make interventions in the world based solely on data. In other words, marking something as an anomaly is only the beginning of a real-life investigation. Since we're implicitly assuming that anomalies are rare, we know the cost of verifying potential anomalies won't be super high if our predictions are accurate enough.


## References

+ But what is the Fourier Transform? A visual introduction: https://www.youtube.com/watch?v=spUNpyF58BY&t=4s
+ Dimensionality Reduction for Matrix- and Tensor-Coded Data [Part 1]: https://www.youtube.com/watch?v=hmmnRF66hOA
+ How to cross-validate PCA, clustering, and matrix decomposition models: https://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
+ Algorithms for calculating variance: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
+ StatQuest: Principal Component Analysis (PCA), Step-by-Step: https://www.youtube.com/watch?v=FgakZw6K1QQ
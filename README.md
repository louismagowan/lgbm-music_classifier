<div id="top"></div>
<!--
*** Copied from https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<h3 align="center">Music Genre Classification with LightGBM</h3>


<!-- ABOUT THE PROJECT -->
## About The Project

This project tries to classify English language songs from a [Kaggle dataset](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs) of publicly available music data into their genres. The pre-processing of the data can be run using the prep_kaggle_data.py script or, alternatively, the processed data is avaialble in the zipped csv in this repo (music_data.csv.zip).

Exploratory data analysis of the music data can be found in the eda.ipynb notebook. Exploratory dimension reduction was also conducted using PCA and t-SNE over the lyric features of the songs (which are highly sparse). This was just to get an early indication of how dimension reduction may be useful to the task of classifying the songs into their genres.

For the actual modelling, the lyric data can be reduced to a user-specified number of output dimensions and can be reduced with either PCA, Truncated SVD or a Keras undercomplete encoder. 

The modelling itself is done using LightGBM and Optuna, a hyperparamter optimization framework.

A range of output dimensions and Optuna trials were tried with the 3 dimension reduction methods. The strongest performing model overall was one that used PCA to reduce the lyric features into 400 dimensions. It achieved a test macro F1 score of 66.48%.

Feel free to check out the Medium article on this [here](https://medium.com/@louismagowan42).
<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [Keras](https://keras.io/)
* [LightGBM]([https://spacy.io/](https://lightgbm.readthedocs.io/en/latest/))
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Optuna](https://optuna.readthedocs.io/en/stable/tutorial/index.html)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

* LinkedIn- [Louis Magowan](https://www.linkedin.com/in/louismagowan/)
* Medium - [Louis Magowan](https://medium.com/@louismagowan42)
* Project Link: [https://github.com/louismagowan/lgbm-music_classifier](https://github.com/louismagowan/lgbm-music_classifier)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [othneildrew - README template](https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md)
* [Bex T. - Medium Writer](https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5)
* [Muhammad Nakhaee, Kaggle](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/louismagowan/lgbm-music_classifier.svg?style=for-the-badge
[contributors-url]: https://github.com/louismagowan/lgbm-music_classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/louismagowan/lgbm-music_classifier.svg?style=for-the-badge
[forks-url]: https://github.com/louismagowan/lgbm-music_classifier/network/members
[stars-shield]: https://img.shields.io/github/stars/louismagowan/lgbm-music_classifier.svg?style=for-the-badge
[stars-url]: https://github.com/louismagowan/lgbm-music_classifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/louismagowan/lgbm-music_classifier.svg?style=for-the-badge
[issues-url]: https://github.com/louismagowan/lgbm-music_classifier/issues
[license-shield]: https://img.shields.io/github/license/louismagowan/lgbm-music_classifier.svg?style=for-the-badge
[license-url]: https://github.com/louismagowan/lgbm-music_classifier/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/louismagowan/
[product-screenshot]: images/screenshot.png

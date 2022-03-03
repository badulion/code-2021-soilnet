<div id="top"></div>


<!-- ABOUT THE PROJECT -->
## About The Project

This is the current code for the newest SoilNet models and experiments

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This describes how to setup the code.

### Prerequisites

To get started you will need a python3 environment and the soil dataset. Copy both the labeled and unlabeled datasets to `dataset/data/%dataset name%` and change the appropriate paths in `config.yaml`

### Installation

To run the experiments you will need the packages listed in `requirements.txt`

install them using
```sh
pip install -r requirements.txt
```
or any other method of your choice.


<!-- USAGE EXAMPLES -->
## Usage

To run the experiments simply run the `main.py` script, overriding the configurations if necessary using hydra syntax, e.g.:

```
python3 main.py model=idw vars=metrical
```




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png

[![MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)](http://www.mbari.org)
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/downloads/)
 
**sinker-shadowgraph** 

This repository processes videos from the ISIIS instrument, it get the frames of the videos, transform the format of the videos from avi to mp4, get the depth for every images based on the CTD of the ROV profile, and other features.

The ISIIS images are like this.
![Alt text](<img/CFE_ISIIS-243-2024-03-19 12-58-20.186_0588 (1).jpg>)


**Installation**

```
conda env create -f environment.yml
conda activate isiis
```

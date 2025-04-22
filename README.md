# prime

Code for "_Prototypical Extreme Multi-label Classification with a Dynamic Margin Loss_" published in NAACL 2025

## Requirements

- [pyxclib](https://github.com/kunaldahiya/pyxclib): for Evaluation and Approximate nearest neighbor search
- [deepxml](https://github.com/kunaldahiya/deepxml-base): PRIME is built on top of the DeepXML framework. This repository uses modules from the deepxml library.

## Setting up

---

### Expected directory structure

```txt
+-- <work_dir>
|  +-- programs
|  |  +-- prime
|  |    +-- prime
|  +-- data
|    +-- <dataset>
|  +-- models
|  +-- results
```

### Download data for PRIME

```txt
* Download the (zipped file) raw data from The XML repository [5].  
* Extract the zipped file into data directory. 
* The following files should be available in <work_dir>/data/<dataset> (create empty filter files if unavailable):
    - trn.json.gz
    - tst.json.gz
    - lbl.json.gz
    - filter_labels_text.txt
    - filter_labels_train.txt
```

### A single learner

The algorithm can be run as follows. A json file (e.g., configs/PRIME/LF-AmazonTitles-131K.json) is used to specify architecture and other arguments. Please refer to the full documentation below for more details.

```bash
python runner.py PRIME ~/Workspace LF-AmazonTitles-131K 0 22
```

## Full Documentation

Please make sure that the required data and config files are available.

### Run PRIME

```txt
python runner.py <method> <work_dir> <dataset> <version> <seed>

* method
  PRIME builds upon NGAME[2], SiameseXML [3] and DeepXML[4] for training. An encoder is trained in M1 and the classifier is trained in M-II.
  - PRIME: The intermediate representation is not fine-tuned while training the classifier (more scalable; suitable for large datasets).

* work_dir
  - The working directory. Please refer to the directory structure above
  - It will dump the model and results under this

* dataset
  - Name of the dataset.
  - configs/PRIME/<dataset>.json must be available as it defines architecture, hyper-parameters
  - PRIME expects the following files in <work_dir>/data/<dataset>
    - filter_labels_test.txt (put empty file or set as null in config when unavailable)
    - trn.json.gz
    - tst.json.gz
    - lbl.json.gz
  - The code will automatically extract and tokenize the data

* version
  - different runs could be managed by version and seed.
  - models and results are stored with this argument.

* seed
  - seed value as used by numpy and PyTorch.
```

## ToDo

- [x] Loss
- [x] Network Architecture
- [x] Network Class
- [x] Model Class
- [x] Runner Code
- [ ] Respect Seed Value
- [x] Dropout for Classification layer
- [ ] Support for csv input files
- [ ] Test

## Cite as

```bib
@InProceedings{Dahiya25,
    author = "Dahiya, K. and Ortego, D. and Jim{\'e}nez, D.",
    title = "Prototypical Extreme Multi-label Classification with a Dynamic Margin Loss",
    booktitle = "NAACL",
    month = "April",
    year = "2025"
}
```

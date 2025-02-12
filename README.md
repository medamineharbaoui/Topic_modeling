# Topic Modeling with LDA and BERTopic

## Project Overview
This project aims to perform topic modeling using **Latent Dirichlet Allocation (LDA)** and **BERTopic** on a dataset consisting of 400 ArXiv document summaries. The results include:
- **10 topics** with **20 words** per topic for each model.
- **Coherence** and **diversity** scores to evaluate the quality of clustering.

## Requirements
- Run **MakeLexiconandTDM.py** to generate necessary datasets.
- Use **LDA-FromPkl.py** for topic modeling using LDA.
- Use **BERTopicFromSubfile.py** for topic modeling using BERTopic.
- Compute **coherence** and **diversity** metrics for each model.
- Process the topic modeling results from Lamirel (ArXiv10010gnge1-tn.fuvp).

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required libraries: `nltk`, `gensim`, `bertopic`, `pandas`, `json`, `multiprocessing`

```bash
pip install nltk gensim bertopic pandas jsonlines
```

### Dataset & Preprocessing
The dataset consists of **400 ArXiv document summaries**. First, we need to generate the lexicon and term-document matrix:

#### Running `MakeLexiconandTDM.py`
**Issue faced:** Missing tagger package in NLTK.

![Error](/SDM_Project_Capture/error.png)

**Solution:** Download and manually install the missing package:
1. Download from: [Averaged Perceptron Tagger](https://github.com/nltk/nltk_data/blob/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip)
2. Extract the files and place them in your NLTK taggers directory.
3. Rename the directory to match the expected path.

**Generated Files:**
```
Lexique400-2.pkl
Lexique400-2.txt
TDM400-2.pkl
TDM400-2.txt
TextsArXiv400.pkl
```
These files were moved to the `output/` folder for further processing.

## Running LDA Topic Modeling
**Modifications made:**
- Renamed the dataset files:
  - `Lexique400-2.pkl` → `LexiqueArXiv400.pkl`
  - `TDM400-2.pkl` → `TDMArXiv400.pkl`
  - `TextsArXiv400.pkl` → `Texts400-2.pkl`
- Added **multiprocessing safeguard** for Windows:
  ```python
  if __name__ == "__main__":
  ```
- **Why?** Windows does not use `fork()` to create subprocesses like Linux/macOS. Instead, it spawns new processes, requiring explicit imports to prevent infinite recursion.
- **Results:**
  - The topic modeling results were saved in `lda_results_ArXiv400.json`.

## Running BERTopic
Similar to LDA, **BERTopicFromSubfile.py** required:
- Adding the **multiprocessing safeguard**.
- Saving the dataset to `dataset.csv`.
- Storing the BERTopic results in `BERTopicFromSubfile_results/top_words.json`.

## Processing Lamirel Results
We processed `ArXiv10010gnge1-tn.fuvp`, extracting only the **first 20 words** of each topic.

## Evaluating Topic Models
We implemented **two key metrics**:
1. **Coherence**: Measures topic quality based on word co-occurrence.
2. **Diversity**: Measures word uniqueness across topics.

Each model's top **20 words** for **10 topics** were used as input, producing **one numeric score per model**:
- LDA: Coherence = `X.XX`, Diversity = `Y.YY`
- BERTopic: Coherence = `A.AA`, Diversity = `B.BB`

## Final Deliverables
- **Topic Modeling Results:**
  - LDA: `lda_results_ArXiv400.json`

  ![LDA_Results](/SDM_Project_Capture/LDA1.png)

  ![LDA_Results](/SDM_Project_Capture/LDA2.png)

  ![LDA_Results](/SDM_Project_Capture/LDA3.png)

  ![LDA_Results](/SDM_Project_Capture/LDA4.png)

  ![LDA_Results](/SDM_Project_Capture/LDA5.png)

  ![LDA_Results](/SDM_Project_Capture/LDA_result.png)

  - BERTopic: `BERTopicFromSubfile_results/top_words.json`

  ![LDA_Results](/SDM_Project_Capture/BERTopic_1.png


## Conclusion
This project successfully applied **LDA** and **BERTopic** for topic modeling, handled **multiprocessing challenges on Windows**, and computed **coherence & diversity scores** to evaluate the models.


# Module for replacing survey text terms with their canonical synonyms

# Standard Python libraries import
from pathlib import Path
import gzip
import shutil
import urllib.request
import urllib.error
import pandas as pd
import nltk

# NLTK elements import
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Third-party libraries import
# Already trained embeddings for spanish (Word2Vec) are used to find synonyms
# for the canonical terms.
from gensim.models import KeyedVectors

from settings import (  SYNONYMS_MODEL_PATH,
                        SYNONYMS_MODEL_URL,
                        canonical_synonyms_seeds,
                        colombian_colloquialisms,
                        SYNONYMS_SIMILARITY_THRESHOLD,
                        SYNONYMS_TOPN
                      )


# Class definition
class SynonymReplacer:
    """
    Replaces terms in a text with their most common synonym, using a
    pretrained Word2Vec model (Spanish) combined with a manually curated
    dictionary of regional colloquialisms.

    This class is independent from the 'Cleaner' class: it operates
    directly on plain text (str) and does not depend on a DataFrame or a
    survey-specific configuration to work.
    """

    def __init__(self,
                 model_path: str | Path = SYNONYMS_MODEL_PATH,
                 model_url: str = SYNONYMS_MODEL_URL,
                 similarity_threshold: float = SYNONYMS_SIMILARITY_THRESHOLD,
                 topn: int = SYNONYMS_TOPN) -> None:
        """
        Initializes the SynonymReplacer class with the specified parameters.

        Args:
            model_path (str | Path): Local path where the pretrained
                Word2Vec model is expected to be found (or downloaded to).
            model_url (str): Remote URL used to download the pretrained
                model if it is not found locally.
            similarity_threshold (float): Minimum cosine similarity value
                required to accept a word as a valid synonym.
            topn (int): Number of candidate words to retrieve from the
                model per canonical seed word.
        """
        self.model_path = Path(model_path)
        self.model_url = model_url
        self.similarity_threshold = similarity_threshold
        self.topn = topn

        self.canonical_synonyms_seeds = canonical_synonyms_seeds
        self.colombian_colloquialisms = colombian_colloquialisms

        self.model = None
        self.synonyms_dict = None

    def _download_model(self) -> None:
        """
        Downloads the pretrained Word2Vec model (.bin.gz) from
        'self.model_url' and decompresses it into 'self.model_path'.

        Raises:
            ValueError: If the download or decompression process fails
                (e.g., connection error, invalid URL, corrupted file).
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        compressed_path = self.model_path.with_suffix(
            self.model_path.suffix + ".gz"
        )

        print(f"Model not found locally. Downloading from "
              f"'{self.model_url}'. This may take a few minutes...")

        try:
            urllib.request.urlretrieve(self.model_url, compressed_path)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise ValueError(
                f"Could not download the Word2Vec model from "
                f"'{self.model_url}': {exc}"
            ) from exc

        try:
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(self.model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except (OSError, gzip.BadGzipFile) as exc:
            raise ValueError(
                f"Could not decompress the downloaded model file "
                f"'{compressed_path}': {exc}"
            ) from exc
        finally:
            # Remove the compressed file regardless of decompression outcome
            compressed_path.unlink(missing_ok=True)

        print(f"Model downloaded and ready at '{self.model_path}'.")

    def load_model(self) -> KeyedVectors:
        """
        Loads the pretrained Word2Vec model for Spanish, downloading it
        first if it is not found in 'self.model_path'.

        Returns:
            KeyedVectors: The loaded gensim KeyedVectors model.

        Raises:
            ValueError: If the model file cannot be downloaded, or if it
                exists but cannot be loaded (e.g., corrupted or invalid
                binary format).
        """
        if not self.model_path.exists():
            self._download_model()

        try:
            self.model = KeyedVectors.load_word2vec_format(
                self.model_path, binary=True
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Error loading the Word2Vec model from "
                f"'{self.model_path}': {exc}"
            ) from exc

        return self.model

    def _expand_seed(self, canonical: str, seed_word: str) -> dict:
        """
        Finds the words most similar to a given seed word using the loaded
        Word2Vec model, and maps each one to its canonical term if it
        passes the configured similarity threshold.

        Args:
            canonical (str): The canonical term that will replace the
                similar words found (e.g. 'bueno').
            seed_word (str): The word used to query the model for similar
                terms (e.g. 'excelente').

        Returns:
            dict: A mapping of {similar_word: canonical} for every
                candidate word that passed the similarity threshold.
        """
        expanded = {}

        if seed_word not in self.model.key_to_index:
            print(f"Warning: seed word '{seed_word}' not found in the "
                  f"model's vocabulary. Skipping.")
            return expanded

        similar_words = self.model.most_similar(seed_word, topn=self.topn)

        for candidate_word, similarity_score in similar_words:
            if similarity_score >= self.similarity_threshold:
                expanded[candidate_word] = canonical

        return expanded

    def build_synonyms_dict(self) -> dict:
        """
        Builds the full synonyms dictionary by combining:
        1. The canonical seed words themselves (identity mapping).
        2. Words automatically found via Word2Vec 'most_similar' for each
           seed word, filtered by the similarity threshold.
        3. Manually curated Colombian colloquialisms, which take priority
           over the automatically generated mappings.

        This method requires 'self.model' to be already loaded (see
        'load_model').

        Returns:
            dict: The complete {variant_word: canonical_word} dictionary,
                stored in 'self.synonyms_dict'.

        Raises:
            ValueError: If the model has not been loaded before calling
                this method.
        """
        if self.model is None:
            raise ValueError(
                "The Word2Vec model has not been loaded. "
                "Call 'load_model' before 'build_synonyms_dict'."
            )

        auto_dict = {}

        for canonical, seed_words in self.canonical_synonyms_seeds.items():
            # Identity mapping: the canonical word maps to itself
            auto_dict[canonical] = canonical

            for seed_word in seed_words:
                expanded = self._expand_seed(canonical, seed_word)
                auto_dict.update(expanded)

        # Manual colloquialisms take priority and are applied last,
        # overriding any conflicting automatic mapping.
        auto_dict.update(self.colombian_colloquialisms)

        self.synonyms_dict = auto_dict

        return self.synonyms_dict

    def replace_synonyms(self, text: str) -> str:
        """
        Replaces every recognized word in the input text with its
        canonical synonym, based on the dictionary built in
        'build_synonyms_dict'.

        Words not present in the synonyms dictionary are left unchanged.
        The comparison is case-insensitive; replaced words are returned
        in lowercase.

        Args:
            text (str): The raw or cleaned text to process.

        Returns:
            str: The text with recognized terms replaced by their
                canonical synonym.

        Raises:
            ValueError: If the synonyms dictionary has not been built yet.
        """
        if not isinstance(text, str):
            return ''

        if self.synonyms_dict is None:
            raise ValueError(
                "The synonyms dictionary has not been built. "
                "Call 'build_synonyms_dict' before 'replace_synonyms'."
            )

        words = nltk.word_tokenize(text)
        replaced_words = [
            self.synonyms_dict.get(word.lower(), word)
            for word in words
        ]

        return ' '.join(replaced_words)

    def synonyms_to_dataframe(self,
                            data: pd.DataFrame,
                            column: str) -> pd.DataFrame:
        """
        Applies 'replace_synonyms' to every cell of a specified column in
        a DataFrame, generating a new column with the replaced text.

        This method is independent from any specific pipeline: it can be
        applied to any DataFrame and column (e.g. the raw key column, or
        the '{key_column}_clean' / '{key_column}_no_stopwords' columns
        produced by the 'Cleaner' class).

        Args:
            data (pd.DataFrame): The DataFrame containing the column to
                process.
            column (str): The name of the column whose text will be
                processed with synonym replacement.

        Returns:
            pd.DataFrame: The same DataFrame with a new column named
                '{column}_synonyms' containing the replaced text.

        Raises:
            ValueError: If the specified column does not exist in the
                DataFrame, or if the synonyms dictionary has not been
                built yet.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the "
                                f"DataFrame.")

        if self.synonyms_dict is None:
            raise ValueError(
                "The synonyms dictionary has not been built. "
                "Call 'build_synonyms_dict' before 'apply_to_dataframe'."
            )

        data[f"{column}_synonyms"] = (
            data[column].apply(lambda x: self.replace_synonyms(x))
        )

        return data

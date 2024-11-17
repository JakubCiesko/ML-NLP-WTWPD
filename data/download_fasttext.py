import argparse 
import requests 
import tqdm
import os

def get_language_url(language: str) -> str:
    url_base = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki." 
    return url_base + language + ".vec"

def download_language_vectors(language:str, output_filepath:str) -> None: 
    url = get_language_url(language)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(output_filepath, "wb") as f, tqdm.tqdm(
            desc=f"Downloading {output_filepath}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Files downloaded and saved to {output_filepath}")
    else: 
        print(f"Failed to download file. Status code: {response.status_code}")

def main():
    parser = argparse.ArgumentParser(description="Download a pre-trained word vector file.")
    parser.add_argument(
        "language", 
        type=str, 
        help="Language abbreviation (e.g., 'en' for English, 'fr' for French)"
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="The local output filename (e.g., 'wiki.en.vec')"
    )
    args = parser.parse_args()
    download_language_vectors(args.language, args.output)


if __name__ == "__main__":
    main()
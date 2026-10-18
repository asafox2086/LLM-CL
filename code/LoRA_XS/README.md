# LoRA-XS source reference

Author repository pinned in source_manifest.json; source/ contains the fetched original and is not run or installed. Reviewed utils/initialization_utils.py, latent_utils.py, svd_utils.py and config. No root LICENSE was found, so original source is retained on the server as a reference, not redistributed in primary Git.

Independent continual coefficient adaptation is under analysis/onevision_directional_coeff_*.py, not an exact author-recipe reproduction or originality claim. The paper uses small Gaussian R; our paired experiment explicitly uses zero R for an identical initial function, common LR/dropout, and one continuing coefficient matrix across19tasks. PiSSA (NeurIPS2024) also motivates pretrained principal directions.

MODEL_NAMES = [
    'evo2_40b',
    'evo2_7b',
    'evo2_40b_base',
    'evo2_7b_base',
    'evo2_1b_base',
]

HF_MODEL_NAME_MAP = {
    'evo2_40b': 'arcinstitute/evo2_40b',
    'evo2_7b': 'arcinstitute/evo2_7b',
    'evo2_40b_base': 'arcinstitute/evo2_40b_base',
    'evo2_7b_base': 'arcinstitute/evo2_7b_base',
    'evo2_1b_base': 'arcinstitute/evo2_1b_base',
}

CONFIG_MAP = {
    'evo2_7b': 'configs/evo2-7b-1m.yml',
    'evo2_40b': 'configs/evo2-40b-1m.yml',
    'evo2_7b_base': 'configs/evo2-7b-8k.yml',
    'evo2_40b_base': 'configs/evo2-40b-8k.yml',
    'evo2_1b_base': 'configs/evo2-1b-8k.yml',
}


def make_phylotag_from_gbif(
        species_name: str,
) -> dict:
    """
    Returns phylogenetic tags for a given species, to get new tags not in the metadata
    """

    import requests
    def get_taxonomy_from_gbif(species_name):
        url = f"https://api.gbif.org/v1/species/match?name={species_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "kingdom": data.get("kingdom"),
                "phylum": data.get("phylum"),
                "class": data.get("class"),
                "order": data.get("order"),
                "family": data.get("family"),
                "genus": data.get("genus"),
                "species": data.get("species")
            }
        else:
            print(f"Could not find taxonomy for {species_name}")

    taxonomy = get_taxonomy_from_gbif(species_name)
    if taxonomy:
        phylo_tag = (
        f'd__{taxonomy["kingdom"]};'
        f'p__{taxonomy["phylum"]};'
        f'c__{taxonomy["class"]};'
        f'o__{taxonomy["order"]};'
        f'f__{taxonomy["family"]};'
        f'g__{taxonomy["genus"]};'
        f's__{taxonomy["species"]}'
    ).upper()
        phylo_tag = '|'+phylo_tag+'|'
    else:
        print(f"Could not find taxonomy for {species_name}")

    return phylo_tag.upper()


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------

import re


def parse_layer_spec(spec: str) -> set[int] | None:
    """Parse layer specification strings into a set of ints or *None* (all).

    Examples
    --------
    >>> parse_layer_spec("26")
    {26}
    >>> parse_layer_spec("0-2")
    {0, 1, 2}
    >>> parse_layer_spec("1,3,5")
    {1, 3, 5}
    >>> parse_layer_spec("all") is None
    True
    """
    spec = spec.strip()
    if spec.lower() in {"all", "*"}:
        return None

    if re.fullmatch(r"\d+", spec):  # single
        return {int(spec)}

    if re.fullmatch(r"\d+-\d+", spec):  # range
        lo, hi = map(int, spec.split("-"))
        if lo > hi:
            lo, hi = hi, lo
        return set(range(lo, hi + 1))

    if re.fullmatch(r"(\d+,)+\d+", spec):  # comma list
        return {int(x) for x in spec.split(",")}

    raise ValueError(f"Unrecognised layer spec: {spec}")


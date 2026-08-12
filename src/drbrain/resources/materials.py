"""Materials-science vocabulary used by concept normalization and filtering.

These word lists were validated in the research pipeline (git-ignored, sources
below) against a ~172k-fulltext materials-science corpus. They are kept here as
data so the main project and the research rules do not drift:

* ``EXCEPT`` / ``NON_MATERIAL`` / ``ELEMENTS`` / ``KNOWN_FORMULA``
  — ``research/scripts/cg_concept_refine_pipeline.py``
* ``EXCEPT`` extras — ``research/scripts/cg_concept_normalize.py``
* ``NON_MATERIAL`` extras — ``research/scripts/cg_concept_clean.py``
"""

from __future__ import annotations

# Plural forms of uncountable/collective nouns that must NOT be singularized
# (domain-standard labels in materials science keep the plural form).
EXCEPT: set[str] = {
    "materials",
    "properties",
    "devices",
    "methods",
    "structures",
    "systems",
    "states",
    "phases",
    "materials science",
    "transitions",
    "mechanisms",
    "dependencies",
    "analyses",
    "series",
    "species",
    "status",
}

# Biomedical / non-materials residue words (materials corpus polluted by other
# domains during open-concept extraction).
NON_MATERIAL: set[str] = {
    "disease",
    "cancer",
    "schizophrenia",
    "clinical",
    "patient",
    "therapy",
    "drug",
    "pharmacokinetic",
    "antibody",
    "virus",
    "bacteria",
    "escherichia",
    "staphylococcus",
    "tuberculosis",
    "pneumonia",
    "epidemic",
    "surgery",
    "hospital",
    "vaccine",
    "cell culture",
    "genome",
    "protein folding",
    "alzheimer",
    "parkinson",
}

# Chemical elements (verbatim from the research pipeline; up to Pu).
ELEMENTS: set[str] = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U Np Pu".split()
)

# Common formulas recognized by the morphological formula check even when their
# token stream does not parse into element symbols (e.g. H2O, O2).
KNOWN_FORMULA: set[str] = {
    "CO2",
    "SO2",
    "NO2",
    "N2O",
    "CaO",
    "FeO",
    "TiO",
    "SiO2",
    "Al2O3",
    "MgO",
    "ZnO",
    "CuO",
    "NiO",
    "H2O",
    "HCl",
    "H2O2",
}

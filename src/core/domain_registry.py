"""
Domain Registry - Extensible system for multi-domain document processing.

Defines domains with their specific schemas, prompts, and enrichment APIs.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Type
from pydantic import BaseModel

from .schemas import ExtractionResult


class DomainType(str, Enum):
    """Supported scientific domains."""
    
    # Aquatic Ecology
    MARINE_ECOLOGY = "marine_ecology"
    FRESHWATER_ECOLOGY = "freshwater_ecology"
    OCEANOGRAPHY = "oceanography"
    
    # Terrestrial Ecology
    TERRESTRIAL_ECOLOGY = "terrestrial_ecology"
    SOIL_SCIENCE = "soil_science"
    
    # Organism-focused
    MICROBIOLOGY = "microbiology"
    MICROBIAL_ECOLOGY = "microbial_ecology"
    GENETICS = "genetics"
    BOTANY = "botany"
    ZOOLOGY = "zoology"
    MYCOLOGY = "mycology"
    PHYCOLOGY = "phycology"  # Algae
    
    # Taxonomic specializations
    ENTOMOLOGY = "entomology"
    ORNITHOLOGY = "ornithology"
    HERPETOLOGY = "herpetology"
    PARASITOLOGY = "parasitology"
    
    # Environmental Sciences
    TOXICOLOGY = "toxicology"
    CONSERVATION = "conservation"
    PALEOECOLOGY = "paleoecology"
    HYDROLOGY = "hydrology"
    CLIMATE_SCIENCE = "climate_science"
    
    # Biomedical Sciences
    PHARMACOLOGY = "pharmacology"
    EPIDEMIOLOGY = "epidemiology"
    NEUROSCIENCE = "neuroscience"
    IMMUNOLOGY = "immunology"
    
    # Organismal Biology
    PHYSIOLOGY = "physiology"
    ETHOLOGY = "ethology"  # Animal behavior
    BIOTIC_INTERACTIONS = "biotic_interactions"
    
    # Earth Sciences
    GEOLOGY = "geology"
    BIOGEOGRAPHY = "biogeography"
    
    # Quantitative / Modeling
    POPULATION_MODELING = "population_modeling"
    NETWORK_ECOLOGY = "network_ecology"
    SPATIAL_ECOLOGY = "spatial_ecology"
    
    # Methodology / Technical
    METHODOLOGY = "methodology"
    REMOTE_SENSING = "remote_sensing"
    BIOINFORMATICS = "bioinformatics"
    
    # AI / Computational
    MACHINE_LEARNING = "machine_learning"
    COMPUTER_VISION = "computer_vision"
    SOUNDSCAPE_ECOLOGY = "soundscape_ecology"
    AI_MODELING = "ai_modeling"
    DEEP_LEARNING = "deep_learning"
    
    # General / Fallback
    GENERAL_ECOLOGY = "general_ecology"
    UNKNOWN = "unknown"


@dataclass
class DomainConfig:
    """Configuration for a specific scientific domain."""
    
    # Identity
    domain_type: DomainType
    display_name: str
    description: str
    
    # Keywords for classification
    keywords: list[str] = field(default_factory=list)
    species_patterns: list[str] = field(default_factory=list)  # Regex patterns
    
    # Prompts
    system_prompt: str = ""
    extraction_prompt_template: str = ""
    
    # Enrichment APIs
    enrichment_apis: list[str] = field(default_factory=list)
    # e.g., ["worms", "fishbase", "gbif"] for marine_ecology
    
    # Entity types to extract (subset of available schemas)
    entity_types: list[str] = field(default_factory=lambda: [
        "species", "measurements", "locations", "temporal", "relations"
    ])
    
    # Confidence threshold for classification
    classification_threshold: float = 0.7
    
    def get_prompt_path(self, prompts_dir: Path) -> Path:
        """Get path to domain-specific prompt file."""
        return prompts_dir / f"{self.domain_type.value}_extraction.txt"


class DomainRegistry:
    """
    Registry of scientific domains with their configurations.
    
    Extensible: add new domains by registering configurations.
    """
    
    _domains: dict[DomainType, DomainConfig] = {}
    _initialized: bool = False
    
    @classmethod
    def register(cls, config: DomainConfig) -> None:
        """Register a domain configuration."""
        cls._domains[config.domain_type] = config
    
    @classmethod
    def get(cls, domain_type: DomainType) -> DomainConfig | None:
        """Get configuration for a domain."""
        cls._ensure_initialized()
        return cls._domains.get(domain_type)
    
    @classmethod
    def get_all(cls) -> list[DomainConfig]:
        """Get all registered domains."""
        cls._ensure_initialized()
        return list(cls._domains.values())
    
    # Fallback domains that should be penalized when specific domains score well
    _FALLBACK_DOMAINS = {DomainType.GENERAL_ECOLOGY, DomainType.UNKNOWN}
    
    @classmethod
    def get_by_keyword(cls, text: str) -> list[tuple[DomainType, float]]:
        """
        Score domains by keyword matching.
        
        Uses weighted matching: multi-word keywords score higher than
        single-word ones (more specific = more discriminating).
        Penalizes fallback domains when specific domains score well.
        
        Returns list of (domain_type, score) sorted by score descending.
        """
        cls._ensure_initialized()
        text_lower = text.lower()
        
        scores = []
        for domain_type, config in cls._domains.items():
            if not config.keywords:
                continue
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for keyword in config.keywords:
                kw_lower = keyword.lower()
                # Multi-word keywords are worth more (more specific)
                word_count = len(kw_lower.split())
                weight = 1.0 + (word_count - 1) * 0.5  # 1.0, 1.5, 2.0...
                total_weight += weight
                
                if kw_lower in text_lower:
                    weighted_score += weight
            
            if total_weight > 0:
                score = weighted_score / total_weight
            else:
                score = 0.0
            
            if score > 0:
                scores.append((domain_type, score))
        
        # Penalize fallback domains if specific domains scored well
        if scores:
            best_specific = max(
                (s for dt, s in scores if dt not in cls._FALLBACK_DOMAINS),
                default=0.0
            )
            
            if best_specific > 0.05:
                # Reduce fallback scores proportionally
                scores = [
                    (dt, s * 0.3) if dt in cls._FALLBACK_DOMAINS else (dt, s)
                    for dt, s in scores
                ]
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Initialize default domains if not already done."""
        if not cls._initialized:
            cls._register_defaults()
            cls._initialized = True
    
    @classmethod
    def _register_defaults(cls) -> None:
        """Register default domain configurations."""
        
        # Marine Ecology (primary domain for EcoloGRAPH)
        cls.register(DomainConfig(
            domain_type=DomainType.MARINE_ECOLOGY,
            display_name="Marine Ecology",
            description="Studies of marine organisms, fisheries, and ocean ecosystems",
            keywords=[
                "fish", "marine", "ocean", "sea", "fishery", "fisheries",
                "coral", "reef", "pelagic", "benthic", "coastal", "trawl",
                "catch", "stock", "spawning", "larvae", "juvenile",
                "plankton", "phytoplankton", "zooplankton",
                "depth", "salinity", "temperature", "ctd",
                "atlantic", "pacific", "mediterranean", "north sea",
                "gadus", "clupea", "sardina", "thunnus", "salmon"
            ],
            enrichment_apis=["worms", "fishbase", "gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations", "traits"]
        ))
        
        # Freshwater Ecology
        cls.register(DomainConfig(
            domain_type=DomainType.FRESHWATER_ECOLOGY,
            display_name="Freshwater Ecology",
            description="Studies of lakes, rivers, and freshwater ecosystems",
            keywords=[
                "freshwater", "lake", "river", "stream", "pond",
                "limnology", "watershed", "aquatic", "wetland",
                "trout", "bass", "pike", "carp", "catfish",
                "invertebrate", "macroinvertebrate", "insect larvae"
            ],
            enrichment_apis=["gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Microbiology
        cls.register(DomainConfig(
            domain_type=DomainType.MICROBIOLOGY,
            display_name="Microbiology",
            description="Studies of microorganisms, bacteria, and microbial communities",
            keywords=[
                "bacteria", "bacterial", "microbe", "microbial", "microbiome",
                "16s", "rrna", "otu", "asv", "amplicon",
                "culture", "colony", "strain", "isolate",
                "antibiotic", "resistance", "pathogen",
                "escherichia", "staphylococcus", "streptococcus"
            ],
            enrichment_apis=["ncbi_taxonomy", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Microbial Ecology
        cls.register(DomainConfig(
            domain_type=DomainType.MICROBIAL_ECOLOGY,
            display_name="Microbial Ecology",
            description="Ecological studies of microbial communities and their interactions",
            keywords=[
                "microbial ecology", "microbiome", "metagenomics", "metatranscriptomics",
                "community composition", "diversity", "richness", "alpha diversity",
                "beta diversity", "functional guild", "nitrogen cycle", "carbon cycle",
                "biogeochemical", "symbiosis", "quorum sensing", "biofilm",
                "soil microbiome", "gut microbiome", "rhizosphere", "holobiont"
            ],
            enrichment_apis=["ncbi_taxonomy", "silva", "greengenes", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Genetics / Genomics
        cls.register(DomainConfig(
            domain_type=DomainType.GENETICS,
            display_name="Genetics & Genomics",
            description="Studies of genes, genomes, and genetic variation",
            keywords=[
                "gene", "genetic", "genome", "genomic", "dna", "rna",
                "sequencing", "snp", "mutation", "allele", "locus",
                "transcriptome", "proteome", "expression",
                "pcr", "primer", "amplification",
                "phylogeny", "phylogenetic", "clade"
            ],
            enrichment_apis=["ncbi_gene", "ensembl", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "relations"]
        ))
        
        # Botany
        cls.register(DomainConfig(
            domain_type=DomainType.BOTANY,
            display_name="Botany",
            description="Studies of plants and plant ecosystems",
            keywords=[
                "plant", "vegetation", "flora", "botanical",
                "tree", "shrub", "herb", "grass", "flower",
                "photosynthesis", "chlorophyll", "stomata",
                "forest", "woodland", "prairie", "meadow",
                "seed", "germination", "pollination"
            ],
            enrichment_apis=["gbif", "ipni", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Zoology
        cls.register(DomainConfig(
            domain_type=DomainType.ZOOLOGY,
            display_name="Zoology",
            description="Studies of animals and animal behavior",
            keywords=[
                "animal", "mammal", "bird", "reptile", "amphibian",
                "behavior", "behaviour", "ethology", "migration",
                "predator", "prey", "carnivore", "herbivore",
                "habitat", "territory", "population", "census"
            ],
            enrichment_apis=["gbif", "iucn", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations", "traits"]
        ))
        
        # Remote Sensing / Methodology
        cls.register(DomainConfig(
            domain_type=DomainType.REMOTE_SENSING,
            display_name="Remote Sensing",
            description="Satellite imagery, spectral analysis, and geospatial methods",
            keywords=[
                "remote sensing", "satellite", "imagery", "spectral",
                "ndvi", "landsat", "sentinel", "modis",
                "classification", "segmentation", "pixel",
                "gis", "geospatial", "raster", "vector"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["locations", "temporal", "measurements"]
        ))
        
        # Methodology (technical papers)
        cls.register(DomainConfig(
            domain_type=DomainType.METHODOLOGY,
            display_name="Methodology",
            description="Methods, algorithms, and technical approaches",
            keywords=[
                "method", "algorithm", "model", "approach",
                "validation", "accuracy", "precision", "recall",
                "machine learning", "deep learning", "neural network",
                "statistical", "analysis", "workflow", "pipeline"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["measurements"]
        ))
        
        # =====================================================
        # NEW DOMAINS
        # =====================================================
        
        # Mycology
        cls.register(DomainConfig(
            domain_type=DomainType.MYCOLOGY,
            display_name="Mycology",
            description="Studies of fungi and fungal ecosystems",
            keywords=[
                "fungus", "fungi", "mushroom", "mycelium", "spore",
                "yeast", "mold", "basidiomycete", "ascomycete",
                "ectomycorrhiza", "arbuscular", "decomposer",
                "fruiting body", "hyphae", "substrate"
            ],
            enrichment_apis=["index_fungorum", "mycobank", "gbif", "crossref"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Phycology (Algae)
        cls.register(DomainConfig(
            domain_type=DomainType.PHYCOLOGY,
            display_name="Phycology",
            description="Studies of algae and algal ecology",
            keywords=[
                "algae", "algal", "seaweed", "diatom", "dinoflagellate",
                "chlorophyte", "rhodophyte", "phaeophyte", "bloom",
                "macroalgae", "microalgae", "kelp", "cyanobacteria",
                "chlorophyll", "primary production"
            ],
            enrichment_apis=["algaebase", "worms", "gbif", "crossref"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Entomology
        cls.register(DomainConfig(
            domain_type=DomainType.ENTOMOLOGY,
            display_name="Entomology",
            description="Studies of insects and arthropods",
            keywords=[
                "insect", "arthropod", "beetle", "butterfly", "moth",
                "bee", "wasp", "ant", "fly", "mosquito", "dragonfly",
                "pollinator", "pest", "larvae", "metamorphosis",
                "exoskeleton", "compound eye", "antenna"
            ],
            enrichment_apis=["gbif", "inaturalist", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations", "traits"]
        ))
        
        # Ornithology
        cls.register(DomainConfig(
            domain_type=DomainType.ORNITHOLOGY,
            display_name="Ornithology",
            description="Studies of birds and avian ecology",
            keywords=[
                "bird", "avian", "ornithology", "migration", "nest",
                "feather", "beak", "wing", "flock", "breeding",
                "songbird", "raptor", "waterfowl", "passerine",
                "clutch", "egg", "chick", "flyway"
            ],
            enrichment_apis=["ebird", "gbif", "xeno_canto", "crossref"],
            entity_types=["species", "measurements", "locations", "temporal", "relations", "traits"]
        ))
        
        # Herpetology
        cls.register(DomainConfig(
            domain_type=DomainType.HERPETOLOGY,
            display_name="Herpetology",
            description="Studies of reptiles and amphibians",
            keywords=[
                "reptile", "amphibian", "snake", "lizard", "turtle",
                "frog", "toad", "salamander", "crocodile", "gecko",
                "ectotherm", "cold-blooded", "scale", "venom",
                "metamorphosis", "tadpole", "basking"
            ],
            enrichment_apis=["amphibiaweb", "reptile_database", "gbif", "crossref"],
            entity_types=["species", "measurements", "locations", "temporal", "relations", "traits"]
        ))
        
        # Parasitology
        cls.register(DomainConfig(
            domain_type=DomainType.PARASITOLOGY,
            display_name="Parasitology",
            description="Studies of parasites and host-parasite interactions",
            keywords=[
                "parasite", "parasitic", "host", "infection", "vector",
                "helminth", "nematode", "trematode", "cestode",
                "protozoan", "malaria", "plasmodium", "tick", "flea",
                "transmission", "prevalence", "intensity"
            ],
            enrichment_apis=["ncbi_taxonomy", "pubmed", "crossref"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Toxicology
        cls.register(DomainConfig(
            domain_type=DomainType.TOXICOLOGY,
            display_name="Toxicology",
            description="Studies of toxins, pollutants, and their effects",
            keywords=[
                "toxin", "toxic", "pollutant", "contaminant", "pesticide",
                "heavy metal", "mercury", "lead", "cadmium", "pcb",
                "bioaccumulation", "biomagnification", "lc50", "ld50",
                "exposure", "dose-response", "ecotoxicology"
            ],
            enrichment_apis=["ecotox", "pubchem", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Conservation
        cls.register(DomainConfig(
            domain_type=DomainType.CONSERVATION,
            display_name="Conservation Biology",
            description="Studies of biodiversity conservation and endangered species",
            keywords=[
                "conservation", "endangered", "threatened", "iucn",
                "protected", "extinction", "invasive", "biodiversity",
                "habitat loss", "fragmentation", "restoration",
                "captive breeding", "reintroduction", "reserve"
            ],
            enrichment_apis=["iucn_red_list", "gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Paleoecology
        cls.register(DomainConfig(
            domain_type=DomainType.PALEOECOLOGY,
            display_name="Paleoecology",
            description="Studies of past ecosystems and paleoclimate",
            keywords=[
                "fossil", "paleoclimate", "quaternary", "pollen",
                "sediment", "core", "holocene", "pleistocene",
                "palaeo", "paleo", "reconstruction", "proxy",
                "isotope", "radiocarbon", "stratigraphy"
            ],
            enrichment_apis=["pbdb", "neotoma", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # Soil Science
        cls.register(DomainConfig(
            domain_type=DomainType.SOIL_SCIENCE,
            display_name="Soil Science",
            description="Studies of soil and edaphic ecosystems",
            keywords=[
                "soil", "edaphic", "rhizosphere", "nitrogen fixation",
                "decomposition", "organic matter", "humus", "clay",
                "nutrient cycling", "microbial biomass", "earthworm",
                "ph", "cation exchange", "texture"
            ],
            enrichment_apis=["isric", "soilgrids", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Hydrology
        cls.register(DomainConfig(
            domain_type=DomainType.HYDROLOGY,
            display_name="Hydrology",
            description="Studies of water systems and hydrological processes",
            keywords=[
                "groundwater", "runoff", "aquifer", "water table",
                "discharge", "watershed", "catchment", "precipitation",
                "evapotranspiration", "streamflow", "flood", "drought"
            ],
            enrichment_apis=["usgs_water", "crossref", "semantic_scholar"],
            entity_types=["measurements", "locations", "temporal"]
        ))
        
        # Pharmacology
        cls.register(DomainConfig(
            domain_type=DomainType.PHARMACOLOGY,
            display_name="Pharmacology",
            description="Studies of drugs and pharmaceutical compounds",
            keywords=[
                "drug", "compound", "dose", "ic50", "ec50", "efficacy",
                "pharmacokinetic", "pharmacodynamic", "metabolism",
                "receptor", "agonist", "antagonist", "binding",
                "clinical trial", "therapeutic"
            ],
            enrichment_apis=["drugbank", "chembl", "pubchem", "pubmed"],
            entity_types=["measurements", "relations"]
        ))
        
        # Epidemiology
        cls.register(DomainConfig(
            domain_type=DomainType.EPIDEMIOLOGY,
            display_name="Epidemiology",
            description="Studies of disease patterns and public health",
            keywords=[
                "outbreak", "prevalence", "incidence", "mortality",
                "epidemic", "pandemic", "transmission", "vector",
                "case-control", "cohort", "risk factor", "surveillance"
            ],
            enrichment_apis=["who", "cdc", "pubmed", "crossref"],
            entity_types=["measurements", "locations", "temporal", "species"]
        ))
        
        # Neuroscience
        cls.register(DomainConfig(
            domain_type=DomainType.NEUROSCIENCE,
            display_name="Neuroscience",
            description="Studies of the brain and nervous system",
            keywords=[
                "brain", "neuron", "synapse", "cortex", "hippocampus",
                "neurotransmitter", "dopamine", "serotonin", "axon",
                "dendrite", "action potential", "fmri", "eeg", "cognition"
            ],
            enrichment_apis=["neuromorpho", "pubmed", "crossref"],
            entity_types=["measurements", "species", "locations"]
        ))
        
        # Immunology
        cls.register(DomainConfig(
            domain_type=DomainType.IMMUNOLOGY,
            display_name="Immunology",
            description="Studies of the immune system",
            keywords=[
                "antibody", "antigen", "immune", "t-cell", "b-cell",
                "cytokine", "interleukin", "vaccine", "immunization",
                "inflammation", "autoimmune", "lymphocyte", "macrophage"
            ],
            enrichment_apis=["iedb", "pubmed", "crossref"],
            entity_types=["measurements", "species", "relations"]
        ))
        
        # Population Modeling
        cls.register(DomainConfig(
            domain_type=DomainType.POPULATION_MODELING,
            display_name="Population Modeling",
            description="Quantitative population dynamics studies",
            keywords=[
                "population dynamics", "matrix model", "leslie",
                "carrying capacity", "growth rate", "density dependence",
                "age structure", "survival", "fecundity", "projection"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "temporal"]
        ))
        
        # Network Ecology
        cls.register(DomainConfig(
            domain_type=DomainType.NETWORK_ECOLOGY,
            display_name="Network Ecology",
            description="Studies of ecological networks and food webs",
            keywords=[
                "network", "node", "edge", "centrality", "food web",
                "connectance", "modularity", "nestedness", "trophic",
                "interaction", "mutualistic", "antagonistic"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["species", "relations", "measurements"]
        ))
        
        # Spatial Ecology
        cls.register(DomainConfig(
            domain_type=DomainType.SPATIAL_ECOLOGY,
            display_name="Spatial Ecology",
            description="Studies of spatial patterns in ecology",
            keywords=[
                "spatial", "home range", "dispersal", "connectivity",
                "metapopulation", "landscape", "corridor", "patch",
                "movement", "telemetry", "gps", "tracking"
            ],
            enrichment_apis=["movebank", "gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # =====================================================
        # FALLBACK DOMAINS
        # =====================================================
        
        # =====================================================
        # AI / COMPUTATIONAL
        # =====================================================
        
        # Machine Learning
        cls.register(DomainConfig(
            domain_type=DomainType.MACHINE_LEARNING,
            display_name="Machine Learning",
            description="Machine learning applied to ecological and biological data",
            keywords=[
                "machine learning", "random forest", "gradient boosting",
                "support vector", "classification", "regression", "clustering",
                "feature extraction", "training", "validation", "test set",
                "cross-validation", "accuracy", "precision", "recall",
                "f1-score", "auc", "roc", "confusion matrix",
                "supervised", "unsupervised", "prediction", "model",
                "xgboost", "lightgbm", "sklearn", "scikit-learn"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # Computer Vision
        cls.register(DomainConfig(
            domain_type=DomainType.COMPUTER_VISION,
            display_name="Computer Vision",
            description="Computer vision for species identification, tracking and monitoring",
            keywords=[
                "computer vision", "image classification", "object detection",
                "image segmentation", "convolutional", "cnn", "yolo",
                "resnet", "vgg", "efficientnet", "image recognition",
                "bounding box", "annotation", "camera trap", "photo-id",
                "species identification", "visual", "pixel", "roi",
                "image processing", "morphometry", "phenotyping",
                "video analysis", "tracking", "mot", "detection"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # Soundscape Ecology / Bioacoustics
        cls.register(DomainConfig(
            domain_type=DomainType.SOUNDSCAPE_ECOLOGY,
            display_name="Soundscape Ecology",
            description="Acoustic monitoring, bioacoustics, and soundscape analysis",
            keywords=[
                "soundscape", "bioacoustics", "acoustic", "audio",
                "spectrogram", "sound", "vocalization", "call", "song",
                "frequency", "acoustic index", "aci", "ndsi", "adi",
                "soundscape ecology", "ecoacoustics", "audio analysis",
                "recording", "microphone", "audiomoth", "song meter",
                "bird song", "whale call", "frog chorus",
                "mel-spectrogram", "mfcc", "sound event detection",
                "acoustic monitoring", "passive acoustic"
            ],
            enrichment_apis=["gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # AI Modeling (General AI/Ecological models)
        cls.register(DomainConfig(
            domain_type=DomainType.AI_MODELING,
            display_name="AI Modeling",
            description="Artificial intelligence models for ecological prediction and analysis",
            keywords=[
                "artificial intelligence", "ai-based", "ai model",
                "neural network", "transformer", "attention mechanism",
                "large language model", "llm", "gpt", "bert",
                "generative", "foundation model", "pre-trained",
                "fine-tuning", "transfer learning", "prompt",
                "species distribution model", "sdm", "maxent",
                "ensemble model", "ecological niche", "habitat suitability",
                "reinforcement learning", "multi-agent"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # Deep Learning
        cls.register(DomainConfig(
            domain_type=DomainType.DEEP_LEARNING,
            display_name="Deep Learning",
            description="Deep learning architectures applied to ecological research",
            keywords=[
                "deep learning", "deep neural network", "dnn",
                "convolutional neural network", "recurrent neural network",
                "lstm", "gru", "autoencoder", "variational",
                "gan", "generative adversarial", "u-net", "encoder-decoder",
                "backpropagation", "epoch", "batch", "dropout",
                "relu", "activation", "pytorch", "tensorflow", "keras",
                "gpu", "cuda", "training loss", "learning rate"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # =====================================================
        # ORGANISMAL BIOLOGY
        # =====================================================
        
        # Physiology
        cls.register(DomainConfig(
            domain_type=DomainType.PHYSIOLOGY,
            display_name="Physiology",
            description="Physiological processes and organismal function",
            keywords=[
                "physiology", "physiological", "metabolism", "metabolic rate",
                "respiration", "oxygen consumption", "heart rate", "blood",
                "thermoregulation", "osmoregulation", "endocrine", "hormone",
                "cortisol", "stress response", "energy budget", "basal metabolic",
                "growth rate", "body condition", "tissue", "organ",
                "enzyme", "protein expression", "cellular", "mitochondria"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal"]
        ))
        
        # Ethology (Animal Behavior)
        cls.register(DomainConfig(
            domain_type=DomainType.ETHOLOGY,
            display_name="Ethology / Animal Behavior",
            description="Animal behavior, movement, and decision-making",
            keywords=[
                "behavior", "behaviour", "behavioral", "behavioural",
                "ethology", "foraging", "mating", "courtship", "display",
                "migration", "movement", "home range", "territory",
                "aggression", "dominance", "social", "group",
                "predator-prey", "anti-predator", "vigilance", "flight response",
                "nesting", "breeding", "parental care", "offspring",
                "circadian", "diel", "activity pattern", "telemetry", "gps tracking"
            ],
            enrichment_apis=["gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Biotic Interactions
        cls.register(DomainConfig(
            domain_type=DomainType.BIOTIC_INTERACTIONS,
            display_name="Biotic Interactions",
            description="Species interactions: predation, competition, mutualism, parasitism",
            keywords=[
                "interaction", "predation", "predator", "prey",
                "competition", "competitive exclusion", "coexistence",
                "mutualism", "symbiosis", "commensalism",
                "herbivory", "grazing", "browsing",
                "pollination", "pollinator", "seed dispersal",
                "parasitism", "host-parasite", "pathogen-host",
                "trophic interaction", "food chain", "food web",
                "interspecific", "intraspecific", "facilitation",
                "keystone species", "ecosystem engineer"
            ],
            enrichment_apis=["gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # =====================================================
        # EARTH SCIENCES
        # =====================================================
        
        # Geology
        cls.register(DomainConfig(
            domain_type=DomainType.GEOLOGY,
            display_name="Geology",
            description="Geological processes and their ecological implications",
            keywords=[
                "geology", "geological", "geomorphology", "lithology",
                "sediment", "sedimentation", "stratigraphy", "rock",
                "mineral", "erosion", "tectonic", "volcanic",
                "substrate", "bedrock", "quaternary", "holocene",
                "glacial", "fluvial", "alluvial", "karst",
                "isotope", "geochemistry", "radiometric"
            ],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["measurements", "locations", "temporal"]
        ))
        
        # Biogeography
        cls.register(DomainConfig(
            domain_type=DomainType.BIOGEOGRAPHY,
            display_name="Biogeography",
            description="Geographic distribution of species and ecological patterns",
            keywords=[
                "biogeography", "biogeographic", "distribution", "range",
                "endemism", "endemic", "cosmopolitan", "disjunct",
                "island biogeography", "species-area", "colonization",
                "dispersal", "vicariance", "refugia", "corridor",
                "latitudinal gradient", "altitudinal gradient", "elevational",
                "biome", "ecoregion", "zoogeographic", "phytogeographic",
                "range shift", "range expansion", "invasive species"
            ],
            enrichment_apis=["gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # =====================================================
        # GENERAL / FALLBACK
        # =====================================================
        
        # General Ecology (fallback - intentionally narrow keywords
        # to avoid absorbing papers that belong to specific domains)
        cls.register(DomainConfig(
            domain_type=DomainType.GENERAL_ECOLOGY,
            display_name="General Ecology",
            description="General ecological studies not fitting specific domains",
            keywords=[
                "general ecology", "ecological theory", "ecosystem services",
                "biodiversity assessment", "species richness", "community ecology",
                "trophic level", "food web", "ecological niche",
                "succession", "disturbance", "resilience"
            ],
            enrichment_apis=["gbif", "crossref", "semantic_scholar"],
            entity_types=["species", "measurements", "locations", "temporal", "relations"]
        ))
        
        # Unknown (fallback)
        cls.register(DomainConfig(
            domain_type=DomainType.UNKNOWN,
            display_name="Unknown Domain",
            description="Unclassified document",
            keywords=[],
            enrichment_apis=["crossref", "semantic_scholar"],
            entity_types=["locations", "temporal"]
        ))


# Convenience function
def get_domain_config(domain_type: DomainType) -> DomainConfig | None:
    """Get configuration for a domain type."""
    return DomainRegistry.get(domain_type)

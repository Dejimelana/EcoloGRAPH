"""
GBIF validate_species method for GBIFOccurrenceClient.

Add this method to src/scrapers/gbif_occurrence_client.py after get_species_key method.
"""

def validate_species(self, scientific_name: str) -> dict | None:
    """
    Validate species name against GBIF Backbone Taxonomy.
    
    Args:
        scientific_name: Scientific name to validate (e.g., "Panthera onca")
        
    Returns:
        Dictionary with validation results:
        {
            "scientific_name": "Panthera onca",
            "canonical_name": "Panthera onca",
            "rank": "SPECIES",
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "family": "Felidae",
            "status": "ACCEPTED",
            "taxon_key": 2435099,
            "matchType": "EXACT" | "FUZZY" | "HIGHERRANK",
            "confidence": 95
        }
        or None if not found.
    """
    self._rate_limit_wait()
    
    try:
        response = self._client.get(
            f"{GBIF_API_BASE}/species/match",
            params={"name": scientific_name, "strict": False}
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Check if we got a valid match
        match_type = data.get("matchType")
        if match_type in ["EXACT", "FUZZY", "HIGHERRANK"]:
            return {
                "scientific_name": data.get("scientificName"),
                "canonical_name": data.get("canonicalName"),
                "rank": data.get("rank"),
                "kingdom": data.get("kingdom"),
                "phylum": data.get("phylum"),
                "class": data.get("class"),
                "order": data.get("order"),
                "family": data.get("family"),
                "genus": data.get("genus"),
                "status": data.get("status"),
                "taxon_key": data.get("usageKey"),
                "matchType": match_type,
                "confidence": data.get("confidence", 0)
            }
        
        return None
        
    except Exception as e:
        logger.error(f"GBIF species validation error for '{scientific_name}': {e}")
        return None

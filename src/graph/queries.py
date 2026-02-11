"""
Cypher queries for common EcoloGRAPH operations.

Pre-built queries for:
- Species lookup and comparison
- Ecological network analysis
- Cross-paper data synthesis
- Spatial queries
"""

# ============================================================
# Species Queries
# ============================================================

SPECIES_PROFILE = """
// Get complete species profile with all related data
MATCH (s:Species {scientific_name: $name})
OPTIONAL MATCH (p:Paper)-[m:MENTIONS]->(s)
OPTIONAL MATCH (s)-[:HAS_MEASUREMENT]->(meas:Measurement)
OPTIONAL MATCH (s)-[r:RELATES_TO]-(other:Species)
RETURN s {
    .*,
    papers: collect(DISTINCT {
        doc_id: p.doc_id, 
        title: p.title, 
        year: p.year
    }),
    measurements: collect(DISTINCT {
        parameter: meas.parameter,
        value: meas.value,
        unit: meas.unit
    }),
    relations: collect(DISTINCT {
        species: other.scientific_name,
        type: r.type,
        direction: CASE WHEN startNode(r) = s THEN 'outgoing' ELSE 'incoming' END
    })
} as profile
"""

SPECIES_COMPARE = """
// Compare measurements between two species
MATCH (s1:Species {scientific_name: $species1})-[:HAS_MEASUREMENT]->(m1:Measurement)
MATCH (s2:Species {scientific_name: $species2})-[:HAS_MEASUREMENT]->(m2:Measurement)
WHERE m1.parameter = m2.parameter
RETURN m1.parameter as parameter,
       s1.scientific_name as species1,
       m1.value as value1,
       m1.unit as unit1,
       s2.scientific_name as species2,
       m2.value as value2,
       m2.unit as unit2
ORDER BY m1.parameter
"""

SPECIES_BY_FAMILY = """
// Get all species in a taxonomic family
MATCH (s:Species {family: $family})
OPTIONAL MATCH (p:Paper)-[:MENTIONS]->(s)
RETURN s.scientific_name as species,
       s.genus as genus,
       count(DISTINCT p) as paper_count
ORDER BY paper_count DESC
"""

SPECIES_CO_OCCURRENCE = """
// Find species that co-occur in papers
MATCH (s1:Species {scientific_name: $name})<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(s2:Species)
WHERE s1 <> s2
RETURN s2.scientific_name as co_occurring_species,
       s2.family as family,
       count(DISTINCT p) as shared_papers
ORDER BY shared_papers DESC
LIMIT 20
"""

# ============================================================
# Ecological Network Queries
# ============================================================

FOOD_WEB = """
// Build food web from predation relationships
MATCH (predator:Species)-[r:RELATES_TO {type: 'predation'}]->(prey:Species)
RETURN predator.scientific_name as predator,
       prey.scientific_name as prey,
       r.description as description,
       r.source_paper as source
"""

ECOLOGICAL_NETWORK = """
// Get full ecological network within N hops of a species
MATCH path = (center:Species {scientific_name: $name})-[:RELATES_TO*1..$depth]-(connected:Species)
WITH nodes(path) as species, relationships(path) as relations
UNWIND relations as r
WITH DISTINCT startNode(r) as from, endNode(r) as to, r.type as type
RETURN from.scientific_name as from_species,
       from.family as from_family,
       to.scientific_name as to_species,
       to.family as to_family,
       type as relation_type
"""

HABITAT_ASSOCIATIONS = """
// Find species associated with a habitat type
MATCH (p:Paper)-[:REFERENCES_LOCATION]->(l:Location {habitat_type: $habitat})
MATCH (p)-[:MENTIONS]->(s:Species)
RETURN s.scientific_name as species,
       s.family as family,
       count(DISTINCT p) as paper_count,
       collect(DISTINCT l.name) as locations
ORDER BY paper_count DESC
"""

# ============================================================
# Cross-Paper Analysis
# ============================================================

MEASUREMENT_SYNTHESIS = """
// Synthesize measurements across papers for a parameter
MATCH (s:Species {scientific_name: $name})-[:HAS_MEASUREMENT]->(m:Measurement {parameter: $parameter})
OPTIONAL MATCH (p:Paper)-[:CONTAINS]->(m)
RETURN m.value as value,
       m.unit as unit,
       m.life_stage as life_stage,
       m.sample_size as n,
       p.title as source,
       p.year as year
ORDER BY p.year
"""

PARAMETER_RANGE = """
// Get min/max range for a measurement parameter across species
MATCH (s:Species)-[:HAS_MEASUREMENT]->(m:Measurement {parameter: $parameter})
RETURN s.scientific_name as species,
       s.family as family,
       min(m.value) as min_value,
       max(m.value) as max_value,
       avg(m.value) as avg_value,
       m.unit as unit
ORDER BY avg_value DESC
"""

RECENT_PAPERS = """
// Get most recent papers for a species
MATCH (p:Paper)-[:MENTIONS]->(s:Species {scientific_name: $name})
RETURN p.doc_id as doc_id,
       p.title as title,
       p.year as year,
       p.doi as doi
ORDER BY p.year DESC
LIMIT 10
"""

# ============================================================
# Spatial Queries
# ============================================================

LOCATION_SPECIES = """
// Get species recorded at a location
MATCH (p:Paper)-[:REFERENCES_LOCATION]->(l:Location)
WHERE l.name CONTAINS $location_name OR l.region CONTAINS $location_name
MATCH (p)-[:MENTIONS]->(s:Species)
RETURN s.scientific_name as species,
       l.name as location,
       l.region as region,
       count(DISTINCT p) as paper_count
ORDER BY paper_count DESC
"""

GEOGRAPHIC_RANGE = """
// Get locations where a species has been studied
MATCH (p:Paper)-[:MENTIONS]->(s:Species {scientific_name: $name})
MATCH (p)-[:REFERENCES_LOCATION]->(l:Location)
WHERE l.latitude IS NOT NULL AND l.longitude IS NOT NULL
RETURN l.name as location,
       l.latitude as lat,
       l.longitude as lon,
       l.country as country,
       p.title as paper,
       p.year as year
"""

NEARBY_STUDIES = """
// Find studies near a geographic point
MATCH (l:Location)
WHERE l.latitude IS NOT NULL AND l.longitude IS NOT NULL
WITH l, 
     point({latitude: l.latitude, longitude: l.longitude}) as loc_point,
     point({latitude: $lat, longitude: $lon}) as search_point
WHERE point.distance(loc_point, search_point) < $radius_km * 1000
MATCH (p:Paper)-[:REFERENCES_LOCATION]->(l)
RETURN l.name as location,
       l.latitude as lat,
       l.longitude as lon,
       point.distance(loc_point, search_point) / 1000 as distance_km,
       p.title as paper,
       p.year as year
ORDER BY distance_km
"""

# ============================================================
# Author & Citation Queries
# ============================================================

AUTHOR_SPECIES = """
// Get species studied by an author
MATCH (a:Author {name: $author_name})<-[:AUTHORED_BY]-(p:Paper)-[:MENTIONS]->(s:Species)
RETURN s.scientific_name as species,
       count(DISTINCT p) as paper_count
ORDER BY paper_count DESC
"""

PROLIFIC_AUTHORS = """
// Find most prolific authors for a species
MATCH (p:Paper)-[:MENTIONS]->(s:Species {scientific_name: $name})
MATCH (p)-[:AUTHORED_BY]->(a:Author)
RETURN a.name as author,
       count(DISTINCT p) as paper_count
ORDER BY paper_count DESC
LIMIT 10
"""

# ============================================================
# Graph Analytics
# ============================================================

SPECIES_CENTRALITY = """
// Find most connected species (simple degree centrality)
MATCH (s:Species)
OPTIONAL MATCH (s)-[r:RELATES_TO]-()
RETURN s.scientific_name as species,
       s.family as family,
       count(r) as connection_count
ORDER BY connection_count DESC
LIMIT 20
"""

BRIDGE_SPECIES = """
// Find species that connect different families (potential keystone species)
MATCH (s1:Species)-[:RELATES_TO]-(bridge:Species)-[:RELATES_TO]-(s2:Species)
WHERE s1.family <> s2.family AND s1.family <> bridge.family
RETURN bridge.scientific_name as bridge_species,
       bridge.family as family,
       count(DISTINCT s1.family) + count(DISTINCT s2.family) as families_connected
ORDER BY families_connected DESC
LIMIT 20
"""

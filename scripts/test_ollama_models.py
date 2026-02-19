"""
Test script to compare Ollama models for EcoloGRAPH extraction.

Compares llama3.2:3b vs qwen2.5-vl:7b-instruct-q6_K_L on:
  1. Entity extraction quality
  2. Speed (tokens/sec)
  3. JSON compliance
  4. Scientific accuracy

Usage:
    python scripts/test_ollama_models.py
"""
import json
import time
import httpx
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

OLLAMA_BASE = "http://localhost:11434/v1"

MODELS = [
    "llama3.2:3b",
    "qwen2.5-vl:7b-instruct-q6_K_L",
]

# Sample ecological text (real paper excerpt style)
SAMPLE_TEXT = """
Abstract: We investigated the population dynamics of Salmo trutta (brown trout) 
and Oncorhynchus mykiss (rainbow trout) in the Ebro River basin, northeastern Spain 
(41.6°N, 0.9°E), between 2018 and 2023. Water temperature ranged from 8.2°C to 22.4°C 
across sampling sites. We collected 847 specimens using electrofishing methodology across 
15 sampling stations at elevations between 450m and 1,200m above sea level.

Brown trout showed a significant decline in abundance (−34%, p<0.01) correlated with 
rising water temperature (mean increase of 1.8°C over the study period). Rainbow trout, 
an invasive species introduced in the 1970s, showed competitive displacement of native 
brown trout in reaches where temperature exceeded 18°C. Body condition factor (K) for 
S. trutta averaged 1.12 ± 0.08, while O. mykiss showed K = 1.28 ± 0.11.

We also documented the presence of Austropotamobius pallipes (white-clawed crayfish, 
IUCN Endangered) in 4 of 15 stations, primarily at higher elevations (>900m). 
The Signal crayfish (Pacifastacus leniusculus), an invasive species, was found in 
8 stations and appears to be displacing the native crayfish through competition 
and transmission of Aphanomyces astaci (crayfish plague).

Key measurements: dissolved oxygen (8.4-12.1 mg/L), pH (7.2-8.1), 
conductivity (185-420 μS/cm), and total phosphorus (0.02-0.15 mg/L).
"""

SYSTEM_PROMPT = """You are an ecological data extraction assistant. You extract structured 
information from scientific papers. Always respond with valid JSON only."""

USER_PROMPT = f"""Extract ecological entities from this text. Return a JSON object with:
{{
    "species": [
        {{"scientific_name": "...", "common_name": "...", "status": "native/invasive/unknown"}}
    ],
    "measurements": [
        {{"parameter": "...", "value": "...", "unit": "...", "species": "..."}}
    ],
    "locations": [
        {{"name": "...", "coordinates": "...", "elevation": "..."}}
    ],
    "relations": [
        {{"source": "...", "target": "...", "type": "...", "description": "..."}}
    ]
}}

TEXT:
{SAMPLE_TEXT}

Return ONLY the JSON object, no additional text.
"""


def test_model(model_name: str) -> dict:
    """Test a single model and return results."""
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {model_name}")
    print(f"{'='*70}")
    
    client = httpx.Client(timeout=120.0)
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    
    # Time the request
    t0 = time.time()
    
    try:
        response = client.post(f"{OLLAMA_BASE}/chat/completions", json=payload)
        response.raise_for_status()
        elapsed = time.time() - t0
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {"model": model_name, "error": str(e)}
    
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    
    # Parse results
    print(f"\n⏱️  Time: {elapsed:.1f}s")
    print(f"📊 Tokens: {usage.get('prompt_tokens', '?')} input, {usage.get('completion_tokens', '?')} output")
    
    if usage.get("completion_tokens"):
        tps = usage["completion_tokens"] / elapsed
        print(f"⚡ Speed: {tps:.1f} tokens/sec")
    
    # Try to parse JSON
    json_valid = False
    extracted = None
    try:
        # Clean up common issues
        clean = content.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        
        extracted = json.loads(clean)
        json_valid = True
        print(f"✅ JSON: Valid")
    except json.JSONDecodeError as e:
        print(f"❌ JSON: Invalid ({e})")
        print(f"   Raw output (first 500 chars):\n   {content[:500]}")
    
    # Evaluate extraction quality
    if extracted:
        species = extracted.get("species", [])
        measurements = extracted.get("measurements", [])
        locations = extracted.get("locations", [])
        relations = extracted.get("relations", [])
        
        print(f"\n📋 Extraction Results:")
        print(f"   🐟 Species: {len(species)}")
        for s in species:
            name = s.get("scientific_name", "?")
            common = s.get("common_name", "?")
            status = s.get("status", "?")
            print(f"      - {name} ({common}) [{status}]")
        
        print(f"   📏 Measurements: {len(measurements)}")
        for m in measurements[:5]:  # Show first 5
            param = m.get("parameter", "?")
            val = m.get("value", "?")
            unit = m.get("unit", "?")
            print(f"      - {param}: {val} {unit}")
        if len(measurements) > 5:
            print(f"      ... and {len(measurements) - 5} more")
        
        print(f"   📍 Locations: {len(locations)}")
        for loc in locations:
            name = loc.get("name", "?")
            coords = loc.get("coordinates", "?")
            print(f"      - {name} ({coords})")
        
        print(f"   🔗 Relations: {len(relations)}")
        for r in relations[:5]:
            src = r.get("source", "?")
            tgt = r.get("target", "?")
            rtype = r.get("type", "?")
            print(f"      - {src} → {tgt} [{rtype}]")
    
    # Score
    score = 0
    if json_valid:
        score += 2
    if extracted:
        # Expected: 5 species (Salmo trutta, O. mykiss, A. pallipes, P. leniusculus, A. astaci)
        expected_species = {"salmo trutta", "oncorhynchus mykiss", "austropotamobius pallipes", 
                          "pacifastacus leniusculus", "aphanomyces astaci"}
        found = {s.get("scientific_name", "").lower() for s in species}
        species_score = len(found & expected_species)
        score += species_score
        
        # Measurements (at least 5 expected)
        score += min(len(measurements), 5)
        
        # Locations
        score += min(len(locations), 2)
        
        # Relations (competition, displacement, disease)
        score += min(len(relations), 3)
    
    max_score = 17  # 2 (json) + 5 (species) + 5 (measurements) + 2 (locations) + 3 (relations)
    
    print(f"\n🏆 SCORE: {score}/{max_score} ({score/max_score*100:.0f}%)")
    
    return {
        "model": model_name,
        "time": elapsed,
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
        "json_valid": json_valid,
        "species_count": len(species) if extracted else 0,
        "measurements_count": len(measurements) if extracted else 0,
        "locations_count": len(locations) if extracted else 0,
        "relations_count": len(relations) if extracted else 0,
        "score": score,
        "max_score": max_score,
        "raw_output": content,
    }


def main():
    print("=" * 70)
    print("🔬 EcoloGRAPH Model Comparison Test")
    print(f"   Models: {', '.join(MODELS)}")
    print(f"   Ollama: {OLLAMA_BASE}")
    print("=" * 70)
    
    # Verify Ollama is running
    try:
        r = httpx.get(f"{OLLAMA_BASE.replace('/v1', '')}/api/tags", timeout=5)
        models_available = [m["name"] for m in r.json().get("models", [])]
        print(f"   Available: {', '.join(models_available)}")
    except Exception as e:
        print(f"❌ Ollama not responding: {e}")
        return
    
    results = []
    for model in MODELS:
        result = test_model(model)
        results.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<40} {'Time':>6} {'Score':>8} {'Species':>8} {'JSON':>6}")
    print(f"{'-'*40} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    
    for r in results:
        model = r["model"][:38]
        time_s = f"{r.get('time', 0):.1f}s"
        score = f"{r.get('score', 0)}/{r.get('max_score', '?')}"
        species = str(r.get("species_count", 0))
        json_ok = "✅" if r.get("json_valid") else "❌"
        print(f"{model:<40} {time_s:>6} {score:>8} {species:>8} {json_ok:>6}")
    
    # Winner
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: (x.get("score", 0), -x.get("time", 999)))
        print(f"\n🏆 Winner: {best['model']} (score: {best['score']}/{best['max_score']})")
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "model_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # Don't save raw_output to keep file small
        clean_results = [{k: v for k, v in r.items() if k != "raw_output"} for r in results]
        json.dump(clean_results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()

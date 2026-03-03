# Bootstrapping-mentes CKKS ML tervezőcsomag

Kutatási prototípus, amely az alábbi 5 modult implementálja:

1. **Mélységbecslő modul** (`fheml/graph.py`) – statikus homomorphic multiplicative depth becslés számítási gráfból.
2. **Aktivációs közelítő modul** (`fheml/activations.py`) – ReLU/Sigmoid/GELU Chebyshev-polinomos közelítése minimális szorzásmélység becsléssel.
3. **Paraméterválasztó modul** (`fheml/params.py`) – automatikus CKKS paramétergenerálás (N és q-lánc) 128-bit biztonsági célra.
4. **Validációs framework** (`fheml/validation.py`) – top-5 accuracy összevetés titkosítatlan és "FHE" logits között.
5. **Benchmark suite** (`benchmarks/run_benchmarks.py`) – LeNet/MobileNet/TinyTransformer referencia benchmark.

> Megjegyzés: ez a kód compiler/runtime planning prototípus. OpenFHE vagy Microsoft SEAL backend bekötése a következő lépés.

## Gyors indulás

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python examples/demo.py
PYTHONPATH=. python benchmarks/run_benchmarks.py
PYTHONPATH=. python benchmarks/validate_requirements.py
PYTHONPATH=. python benchmarks/external_compare.py
```

## Követelmény-ellenőrzés

A `benchmarks/validate_requirements.py` összesíti a minimális elfogadási feltételeket:

- legalább **2** bootstrapping-mentes referencia modell szimulált validációja,
- top-5 accuracy relatív eltérés **< 0.1%**,
- minimális speedup **>= 5x** baseline-hoz képest,
- biztonsági cél **128 bit**.

A script CSV-szerű eredményt ad vissza és `passed=True` esetén minden fenti feltétel teljesül.


## Külső backend összehasonlítás (EVA/CHET, nGraph-HE, Concrete ML, TenSEAL)

A `benchmarks/external_compare.py` script a kért rendszereket ellenőrzi a futtatási környezetben, és összehasonlító riportot ad:

- ha egy backend modul telepítve van, azt jelzi (és itt köthető be a modell-futtató adapter),
- ha nincs telepítve, explicit `unavailable_or_not_measured` státuszt ad.

> Fontos: ebben a környezetben csak azt tudjuk hitelesen állítani, amit ténylegesen futtattunk.
> Az 5x gyorsulás top-tier rendszerekhez képest csak akkor jelenthető ki, ha ezek a backendek ténylegesen telepítve és ugyanazon modelleken mérve vannak.

## Backend integrációs terv

- Fordítási terv exportálása JSON-ba (`CompiledPlan` -> JSON)
- OpenFHE/SEAL adapter réteg a `CKKSParameters` felhasználásával
- Slot packing optimalizáció és rotációs kulcs menedzsment
- Valós modell futtatás legalább 2 referencia hálón bootstrapping nélkül

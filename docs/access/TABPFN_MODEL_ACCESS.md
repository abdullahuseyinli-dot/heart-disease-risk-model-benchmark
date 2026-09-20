# TabPFN model-access record

Package installation date: 2026-08-23
Installed package: `tabpfn==8.4.0`

The default TabPFN v3 weights require the repository owner to accept the Prior
Labs License 1.2 and authenticate. The owner completed that step manually. A credential is
present in the standard user-level TabPFN cache, outside this repository; the
credential value and any derived fingerprint are deliberately not recorded here.

The licence API reports acceptance for `tabpfn_3`, and a 20-row v3 inference smoke
test completed on the NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU with
finite probabilities in `[0, 1]`. This verifies access and execution only; it is
not benchmark evidence.

The source-only studies keep the public v2 checkpoint and authenticated v3 model
as separately named baselines. Package version, resolved configuration, hardware,
and run manifest are recorded for each. V2 and v3 results must never be labelled
interchangeably. No credential, `.env` file, or auth-token file may be committed.

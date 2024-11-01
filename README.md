# ParaAttention

Context parallel attention that works with torch.compile

This aims to include:

- [ ] The fastest accurate attention implemented in Triton, running 50% faster than the originial FA2 implementation on RTX 4090.
- [x] The ability to run context parallel attention in a unified interface, as well as keeping the maximum performance while working with `torch.compile`

# GitHub 업로드용 패키지 (sieve-mask spectral peaks artifact)

이 폴더는 “Project Link” 전체가 아니라, **공표/재현에 필요한 최소 파일만** 모아둔 업로드 번들이다.  
※ 이 결과는 **리만가설(RH) 증명/반증이 아니라**, 실험에서 관측된 “반복 주파수 피크”의 생성 메커니즘(아티팩트) 규명이다.

## 빠른 시작(Windows PowerShell)

```powershell
python -m pip install -r .\requirements.txt
```

```powershell
# (선택) 원인추적(체 단계별)
python .\tools\riemann_experiment.py --mode sieve_sweep --N 2000000 --M 3000000 --Ws 30 210 --windows hann --axes k --L 65536 --num-windows 4 --K 30 --f-min-ratio 0.01 --track-delta-bins 4 --perms 10 --shuffle-block 2048 --seed 42 --out .\reports\riemann_sieve_sweep.md

# mask FFT(마스크 DFT로 피크/랭크)
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 25 --f-min-ratio 0.01 --seed 42 --out .\reports\riemann_mask_fft.md

# 10차(룬 교체/제거 + blur)
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 25 --mask-ablate --mask-swap-last --mask-blur-block 1024 --f-min-ratio 0.01 --seed 42 --out .\reports\riemann_mask_fft_10th.md

# blur 스케일 스윕(임계 경계 스케일 요약표)
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 10 --mask-blur-sweep 32 64 128 256 512 1024 2048 --f-min-ratio 0.01 --seed 42 --out .\reports\riemann_mask_fft_blur_sweep.md
```

## 포함 파일

- `PUBLIC_NOTE.md`: 공표문 초안(복붙용)
- `reports/riemann_sieve_sweep.md`: sieve sweep 결과
- `reports/riemann_mask_fft.md`: 마스크 DFT 결과
- `reports/riemann_mask_fft_10th.md`: 룬 교체/제거 + blur 결과
- `reports/riemann_mask_fft_blur_sweep.md`: blur 스케일 스윕(+요약표)
- `tools/riemann_experiment.py`: 재현 스크립트


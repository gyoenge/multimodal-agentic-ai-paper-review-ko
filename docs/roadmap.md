# 리뷰 로드맵

리뷰 예정 논문 전체 목록입니다. 각 논문 제목을 클릭하면 리뷰 페이지로 이동합니다.

**범례**: ⬜ 예정 &nbsp;|&nbsp; 🔄 진행중 &nbsp;|&nbsp; ✅ 완료

---

### VLM (Vision-Language Model)

VLM 핵심 논문

| 상태 | 논문 | 기관 | 발표 | 한 줄 요약 |
|:----:|------|------|------|------------|
| ⬜ | [CLIP](reviews/vlm/clip.md) | OpenAI | ICML 2021 | 대규모 이미지-텍스트 대조 학습으로 zero-shot 전이 |
| ⬜ | [ALIGN](reviews/vlm/align.md) | Google | ICML 2021 | noisy 웹 데이터로 image-text 표현 학습 스케일업 |
| ⬜ | [Flamingo](reviews/vlm/flamingo.md) | DeepMind | NeurIPS 2022 | frozen LM에 visual 입력 연결, few-shot VLM |
| ⬜ | [BLIP](reviews/vlm/blip.md) | Salesforce | ICML 2022 | 캡션 생성으로 noisy 데이터 부트스트랩 |
| ⬜ | [BLIP-2](reviews/vlm/blip2.md) | Salesforce | ICML 2023 | Q-Former로 frozen 이미지 인코더 + frozen LLM 연결 |
| ⬜ | [LLaVA](reviews/vlm/llava.md) | - | NeurIPS 2023 | GPT-4 생성 instruction data로 visual tuning |
| ⬜ | [InstructBLIP](reviews/vlm/instructblip.md) | Salesforce | NeurIPS 2023 | instruction-aware Q-Former |
| ⬜ | [GPT-4V](reviews/vlm/gpt4v.md) | OpenAI | 2023 | 상용 최강 VLM, 멀티모달 추론 |

VLM 최신 논문 (~2025)

| 상태 | 논문 | 기관 | 연도 | 특징 |
|:----:|------|------|------|------|
| ⬜ | [LLaVA-1.5](reviews/vlm/llava-1-5.md) | - | 2024 | MLP connector, 고해상도 타일링 |
| ⬜ | [InternVL2](reviews/vlm/internvl2.md) | Shanghai AI Lab | 2024 | 오픈소스 SOTA, 동적 해상도 |
| ⬜ | [Qwen2-VL](reviews/vlm/qwen2-vl.md) | Alibaba | 2024 | Naive Dynamic Resolution, 영상 이해 강화 |
| ⬜ | [PaliGemma](reviews/vlm/paligemma.md) | Google | 2024 | SigLIP + Gemma, 경량 VLM |
| ⬜ | [Molmo](reviews/vlm/molmo.md) | AI2 | 2024 | pointing 능력, 오픈소스 |
| ⬜ | [Cambrian-1](reviews/vlm/cambrian-1.md) | NYU | 2024 | spatial vision aggregator, 다중 인코더 |
| ⬜ | [Janus](reviews/vlm/janus.md) | DeepSeek | 2024 | 이해/생성 분리 아키텍처 |
| ⬜ | [Qwen2.5-VL](reviews/vlm/qwen2-5-vl.md) | Alibaba | 2025 | 문서·영상 이해 최강급 |
| ⬜ | [Gemini 2.0](reviews/vlm/gemini2.md) | Google | 2025 | 멀티모달 네이티브, 실시간 처리 |

---

### LLM (Large Language Model)

LLM 핵심 논문

| 상태 | 논문 | 기관 | 발표 | 한 줄 요약 |
|:----:|------|------|------|------------|
| ✅ | [Transformer](reviews/llm/transformer.md) | Google | NeurIPS 2017 | Transformer 아키텍처 제안 |
| ⬜ | [GPT-3](reviews/llm/gpt3.md) | OpenAI | NeurIPS 2020 | 175B 파라미터, few-shot learning |
| ⬜ | [T5](reviews/llm/t5.md) | Google | JMLR 2020 | 모든 NLP 태스크를 text-to-text로 통일 |
| ⬜ | [InstructGPT](reviews/llm/instructgpt.md) | OpenAI | NeurIPS 2022 | RLHF로 instruction following 정렬 |
| ⬜ | [Chain-of-Thought](reviews/llm/chain-of-thought.md) | Google | NeurIPS 2022 | 단계적 추론으로 복잡한 문제 해결 |
| ⬜ | [LLaMA](reviews/llm/llama.md) | Meta | 2023 | 효율적 오픈소스 LLM의 기준점 |
| ⬜ | [LLaMA 2](reviews/llm/llama2.md) | Meta | 2023 | RLHF 포함 chat 버전 공개 |
| ⬜ | [Mistral 7B](reviews/llm/mistral-7b.md) | Mistral AI | 2023 | GQA + sliding window attention |

LLM 최신 논문 (~2025)

| 상태 | 논문 | 기관 | 연도 | 특징 |
|:----:|------|------|------|------|
| ⬜ | [LLaMA 3](reviews/llm/llama3.md) | Meta | 2024 | 405B 공개, 멀티링구얼 강화 |
| ⬜ | [Qwen2.5](reviews/llm/qwen2-5.md) | Alibaba | 2024 | 수학·코드 특화, 72B 오픈소스 |
| ⬜ | [DeepSeek-V3](reviews/llm/deepseek-v3.md) | DeepSeek | 2024 | MoE 671B, 학습 비용 혁신 |
| ⬜ | [Gemma 2](reviews/llm/gemma2.md) | Google | 2024 | knowledge distillation, 경량 고성능 |
| ⬜ | [Phi-3](reviews/llm/phi3.md) | Microsoft | 2024 | 소형 모델 SOTA, textbook 데이터 |
| ⬜ | [o1](reviews/llm/o1.md) | OpenAI | 2024 | test-time compute scaling, 추론 강화 |
| ⬜ | [DeepSeek-R1](reviews/llm/deepseek-r1.md) | DeepSeek | 2025 | 순수 RL로 추론 능력 학습, o1급 오픈소스 |
| ⬜ | [Kimi k1.5](reviews/llm/kimi-k1-5.md) | Moonshot | 2025 | long CoT + 멀티모달 추론 |
| ⬜ | [Llama 4](reviews/llm/llama4.md) | Meta | 2025 | MoE 네이티브, 멀티모달 통합 |

---

### VLA (Vision-Language-Action)

VLA 핵심 논문

| 상태 | 논문 | 기관 | 발표 | 한 줄 요약 |
|:----:|------|------|------|------------|
| ⬜ | [Gato](reviews/vla/gato.md) | DeepMind | 2022 | 하나의 트랜스포머로 600+ 태스크 |
| ⬜ | [SayCan](reviews/vla/saycan.md) | Google | CoRL 2022 | LLM을 affordance로 grounding |
| ⬜ | [RT-1](reviews/vla/rt1.md) | Google | 2022 | 대규모 로봇 데이터로 학습한 Robotics Transformer |
| ⬜ | [PaLM-E](reviews/vla/palm-e.md) | Google | ICML 2023 | 562B embodied multimodal LM |
| ⬜ | [RT-2](reviews/vla/rt2.md) | Google DeepMind | CoRL 2023 | VLM 웹 지식을 로봇 제어로 직접 전이 |

VLA 최신 논문 (~2025)

| 상태 | 논문 | 기관 | 연도 | 특징 |
|:----:|------|------|------|------|
| ⬜ | [Open X-Embodiment](reviews/vla/open-x-embodiment.md) | 여러 기관 | 2023 | 22개 로봇 데이터셋 통합 |
| ⬜ | [Octo](reviews/vla/octo.md) | UC Berkeley | 2024 | 오픈소스 제너럴리스트 로봇 정책 |
| ⬜ | [SpatialVLM](reviews/vla/spatialvlm.md) | Google | CVPR 2024 | 3D 공간 추론 VLM |
| ⬜ | [OpenVLA](reviews/vla/openvla.md) | Stanford | 2024 | 오픈소스 VLA, 7B LLaMA 기반 |
| ⬜ | [π0](reviews/vla/pi0.md) | Physical Intelligence | 2024 | flow matching 기반 VLA, 범용 조작 |
| ⬜ | [CogACT](reviews/vla/cogact.md) | 清华大学 | 2024 | LLM action head로 연속 제어 |
| ⬜ | [RoboVLMs](reviews/vla/robovlms.md) | - | 2024 | VLM을 로봇 정책에 적용하는 체계적 연구 |
| ⬜ | [π0.5](reviews/vla/pi0-5.md) | Physical Intelligence | 2025 | 언어 지시 기반 범용 로봇 |
| ⬜ | [Helix](reviews/vla/helix.md) | Figure AI | 2025 | 인간형 로봇을 위한 VLA |

---

### Agentic AI

Agent 핵심 논문

| 상태 | 논문 | 기관 | 발표 | 한 줄 요약 |
|:----:|------|------|------|------------|
| ⬜ | [ReAct](reviews/agent/react.md) | Princeton / Google | ICLR 2023 | Reasoning + Acting 결합, LLM 에이전트 기반 |
| ⬜ | [Toolformer](reviews/agent/toolformer.md) | Meta | NeurIPS 2023 | LLM이 스스로 툴 사용법을 학습 |
| ⬜ | [Generative Agents](reviews/agent/generative-agents.md) | Stanford | UIST 2023 | 25개 AI 캐릭터가 사회적 행동 시뮬레이션 |
| ⬜ | [Voyager](reviews/agent/voyager.md) | NVIDIA et al. | 2023 | Minecraft에서 LLM 기반 오픈엔디드 에이전트 |
| ⬜ | [AutoGen](reviews/agent/autogen.md) | Microsoft | 2023 | 멀티 에이전트 대화 프레임워크 |
| ⬜ | [AgentBench](reviews/agent/agentbench.md) | 清华大学 | ICLR 2024 | LLM을 에이전트로 평가하는 종합 벤치마크 |

Agent 최신 논문 (~2025)

| 상태 | 논문 | 기관 | 연도 | 특징 |
|:----:|------|------|------|------|
| ⬜ | [WebArena](reviews/agent/webarena.md) | CMU et al. | 2024 | 실제 웹 환경 에이전트 벤치마크 |
| ⬜ | [OSWorld](reviews/agent/osworld.md) | 여러 기관 | 2024 | GUI 컴퓨터 조작 에이전트 벤치마크 |
| ⬜ | [CogAgent](reviews/agent/cogagent.md) | 清华大学 | CVPR 2024 | GUI 이해·탐색 특화 VLM |
| ⬜ | [SWE-agent](reviews/agent/swe-agent.md) | Princeton | 2024 | 소프트웨어 엔지니어링 에이전트, SWE-bench |
| ⬜ | [OpenHands](reviews/agent/openhands.md) | - | 2024 | 오픈소스 AI 소프트웨어 개발 에이전트 |
| ⬜ | [Claude Computer Use](reviews/agent/computer-use.md) | Anthropic | 2024 | 화면·키보드·마우스 직접 조작 |
| ⬜ | [Browser Use](reviews/agent/browser-use.md) | - | 2024 | LLM 기반 웹 브라우저 자동화 |
| ⬜ | [Agentless](reviews/agent/agentless.md) | UIUC | 2024 | 에이전트 없이 SWE-bench 해결 |
| ⬜ | [Manus](reviews/agent/manus.md) | - | 2025 | 범용 자율 에이전트 |


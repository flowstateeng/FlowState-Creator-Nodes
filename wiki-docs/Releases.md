# 🎉 Releases - Current v1.0.0

This page documents the major milestones and releases for the FlowState Creator Nodes project.

---

### 🤏 Minor Release - Small FlowState Chef & WAN Studio Updates
**Date:** 2025-10-01
* Added Tiled Decoding to WAN Studio to help the decoding bottleneck with larger videos.

* Updated the Chef stage execution pipeline to make the loading a little more efficient.

---

### 🔥 Major Release - 🌊👩🏻‍🍳 FlowState Chef & Chef Ingredients!
**Date:** 2025-09-30
Introducing the 🌊👩🏻‍🍳 FlowState Chef and 🌊🥗 FlowState Chef Ingredients nodes! This is a revolutionary two-stage pipeline designed for powerful, instruction-based image editing and refinement. It combines the advanced image editing capabilities of Qwen with the high-quality refinement of FLUX into a single, cohesive, and incredibly efficient workflow.

* Instruction-Based Image Editing: Use the new dual-prompt system to give direct commands to the Qwen model (e.g., "place the woman on the left, change the man's shirt to blue") and then use a separate, descriptive prompt to guide the final FLUX refinement.

* Multi-Image Composition: The Chef Ingredients node allows you to input up to four separate images, which are automatically stitched together into a single composition, making it easy to create scenes with multiple subjects.

* Intelligent Stage Caching: The Chef node is built for efficiency. It intelligently detects which parameters have changed between runs. If you only adjust a setting in the FLUX refinement stage, the node will skip re-running the initial Qwen edit, saving significant generation time.

* Total Creative Control: As with all FlowState nodes, you have granular control over every aspect of the pipeline, including models, LoRAs, samplers, schedulers, and advanced settings for both the Qwen and FLUX stages.

---

### 🔥 Major Release - 🌊🎬 FlowState WAN Studio!
**Date:** 2025-09-22

Introducing 🌊🎬 FlowState WAN Studio, the new flagship node for the FlowState Creator Suite! This is our most advanced node yet, designed to be an all-in-one, comprehensive pipeline for generating high-quality video. It encapsulates a complex, multi-stage workflow into a single, convenient, and powerful interface, saving you time and graph complexity.

* Dual-Model Sampling Pipeline: Utilize two separate UNET models in a sequential workflow. The node intelligently splits the sampling steps, allowing you to use one model for the initial high-noise steps and a second, specialized model for refining low-noise details.

* Advanced LoRA & Model Patching: Get granular control over your model augmentations. Apply separate optimization LoRAs to the high-noise and low-noise stages, patch in an overall style LoRA, and enable Sage Attention, all from within the node.

* Flexible Video Sizing: Easily define your output resolution. Choose from a list of pre-selected aspect ratios, enter custom dimensions, or have the node automatically inherit the resolution from an optional starting frame.

* Complete, Self-Contained Workflow: WAN Studio handles the entire process from start to finish. It manages loading models, encoding prompts, creating the initial latent, patching augmentations, running the two-stage sampling, and decoding the final video.

* Seamless Image-to-Video: By providing an image to the optional starting_frame input, you can guide the video generation process, turning a static image into a dynamic clip with ease.

---

### 🤏 Minor Importing & Patching Updates
**Date:** 2025-09-17

Updates to the System Initialization & model loading/patching routines, as well as to the import pipelines.
* Makes the System Initialization ~0.02s faster.
* More sophisticated model patching strategy results in better model persistence, which leads to fewer model loads.

---

### 🧠 Sage Attention Fix for 🌊🚒 FlowState Flux Engine
**Date:** 2025-09-16

Introduced a more flexible strategy for checking the availability of KJNodes, due to varying naming schemas, depending on how the use installed KJNodes.

---

### 🎨 LoRA Integration for 🌊🚒 FlowState Flux Engine
**Date:** 2025-09-16

Now added `Flux LoRA` styling for the 🌊🚒 FlowState Flux Engine. Simply select `model` & `strength`, or select `none.`

---

### 📖 Wiki Documentation Created
**Date:** 2025-09-16

The official wiki documentation for the project was created, providing users with a central place to find information about the nodes and their usage. The `Nodes.md` file was also updated with detailed input/output information.

---

### ✅ Checkpoint Integration & Media Updates
**Date:** 2025-09-16

This update brought checkpoint integration to the `🌊🚒 FlowState Flux Engine`, allowing for use of bundled checkpoints (e.g., Flux Dev.1 fp8). It also included new media for the Comfy Registry.

---

### 🧠 Sage Attention Integration
**Date:** 2025-09-15

A significant upgrade to the `🌊🚒 FlowState Flux Engine`, integrating `Sage Attention` for more precise and context-aware image manipulation.

---

### 🌊🚒 FlowState Flux Engine - Initial Release
**Date:** 2025-09-15

The initial release of the `🌊🚒 FlowState Flux Engine`, a powerful tool for transforming and manipulating latent images. This release laid the groundwork for future integrations and features.

---

### 🌊🌱 FlowState Latent Source - Stable Release
**Date:** 2025-09-15

The first major node release since the clean slate initiative. The `🌊🌱 FlowState Latent Source` node provides a stable and reliable way to generate latent images for your workflows.

---

### 🚀 Clean slate initiative
**Date:** 2025-09-14

This commit marked a fresh start for the repository, renewing its focus and introducing a new direction for the project. The initiative cleared out old work to make way for a new set of powerful and efficient nodes.

---

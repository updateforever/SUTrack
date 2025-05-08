from typing import List, Dict


def build_prompt(gt: dict, candidates: List[dict]) -> str:
    """
    English single-turn prompt for structured comparison output in JSON format.
    """
    return (
        "The target object has been clearly defined using the bounding box below. "
        "Your task is to generate a structured comparison that distinguishes the **target** from the given **distractor objects**.\n\n"
        "Please DO NOT attempt to guess which is the target — assume the target is already known.\n\n"
        "Compare the target and distractors from the following four aspects:\n"
        "1. Appearance (e.g., color, texture, shape, posture)\n"
        "2. Spatial position (e.g., location in the image, relative layout)\n"
        "3. Behavioral or functional cues (e.g., motion, usage)\n"
        "4. Confusable points (e.g., what makes them hard to distinguish, and how to resolve it)\n\n"
        f"Target bounding box: {gt}\n"
        f"Distractor boxes: {candidates}\n\n"
        "Please return your result strictly in the following JSON format:\n\n"
        "```\n"
        "{\n"
        "  \"appearance\": \"...\",\n"
        "  \"position\": \"...\",\n"
        "  \"interaction\": \"...\",\n"
        "  \"confusable_points\": \"...\",\n"
        "  \"conclusion\": \"...\"\n"
        "}\n"
        "```"
    )
 
def build_prompt_cn(gt: dict, candidates: List[dict], category: str = None) -> str:
    """
    构建完整描述任务的中文 Prompt：
    - 输入图像已标出目标（绿框）与干扰物（红框）；
    - 模型需分析外观、位置、动作、干扰关系；
    - 输出结构化的一句话：[位置特征]的[外观主体]正在[动作状态]，[相似物体指引]。
    """
    category_str = f"该图像中的目标类别为“{category}”，" if category else ""
    return (
        f"{category_str}图中目标物体已用**绿色边框**标出，干扰物体用**红色边框**标出。\n\n"
        f"目标框坐标：{gt}\n"
        f"干扰框列表：{candidates}\n\n"
        "请你结合图像内容，从以下三个方面分析该目标：\n"
        "1. **外观主体**：目标自身的颜色、材质、形状等易于识别的特征；\n"
        "2. **位置与动作状态**：目标在图像中的相对位置（居中、靠边、远近）及其姿态、动作（静止、行走、朝向等）；\n"
        "3. **相似物体排除提示**：目标与周围干扰物在空间或结构上的细微区别，能帮助避免误识。\n\n"
        "🧩 输出要求：请仅输出一句完整的中文描述，句式如下：\n"
        "**[位置特征]的[外观主体]正在[动作状态]，[相似物体指引]。**\n"
        "确保内容连贯、精炼，并具备实际的识别区分价值。"
    )

def build_prompt_level1(gt: dict, candidates: List[dict]) -> str:
    """
    English: Focus on appearance-based differences.
    """
    return (
        "You are given a target object and several distractor objects in an image.\n"
        "Focus specifically on **appearance-based differences** to describe how the target stands out.\n"
        "Appearance may include: color, texture, shape, posture, or unique marks (e.g., labels, stripes).\n\n"
        f"Target box: {gt}\n"
        f"Distractor boxes: {candidates}\n"
        "Output a description (in English) that highlights how the target looks different from the distractors."
    )


def build_prompt_level2(gt: dict, candidates: List[dict]) -> str:
    """
    English: Focus on spatial position relationships.
    """
    return (
        "Compare the **spatial positions** of the target and distractor objects in the image.\n"
        "For example: Is the target near the center or the edge? Is it above or below others? Closer or farther?\n\n"
        f"Target box: {gt}\n"
        f"Distractor boxes: {candidates}\n"
        "Provide a spatial-position-based description that helps localize and distinguish the target."
    )


def build_prompt_level3(gt: dict, candidates: List[dict]) -> str:
    """
    English: Analyze confusing points and how to distinguish.
    """
    return (
        "Analyze the **potential sources of confusion** between the target and distractor objects.\n"
        "What similarities might make them hard to tell apart? What small but crucial details help distinguish them?\n"
        "You may use appearance, position, or behavior for your reasoning.\n\n"
        f"Target box: {gt}\n"
        f"Distractor boxes: {candidates}\n"
        "Return a descriptive analysis that clarifies how to avoid misidentifying the target."
    )


def build_prompt_multi(gt: dict, candidates: List[dict]) -> Dict[str, str]:
    """
    Multi-round prompt builder for level_1 (appearance), level_2 (position), and level_3 (confusable points).
    """
    return {
        "level_1": build_prompt_level1(gt, candidates),
        "level_2": build_prompt_level2(gt, candidates),
        "level_3": build_prompt_level3(gt, candidates)
    }


def build_prompt_level1_cn(gt: dict, candidates: List[dict], category: str = None) -> str:
    category_str = f"待跟踪目标类别为“{category}”，" if category else ""
    return (
        f"你正在查看一幅图像，其中的待跟踪目标物体已用绿色边界框标出，{category_str}干扰物体则以红色框标出。\n\n"
        f"目标框坐标为：{gt}\n"
        f"干扰物框列表为：{candidates}\n\n"
        "请你观察该图像，描述目标的**外观特征**，例如颜色、体型、姿态等，"
        "突出其具备辨识度的可见特征。\n\n"
        "⚠️ 输出要求：请仅输出一句简洁明了的中文描述，用于帮助他人快速识别图中待跟踪目标。"
    )

def build_prompt_level2_cn(gt: dict, candidates: List[dict], category: str = None) -> str:
    category_str = f"待跟踪目标类别为“{category}”，" if category else ""
    return (
        f"你正在查看一幅图像，其中的待跟踪目标物体已用绿色边界框标出，{category_str}干扰物体则以红色框标出。\n\n"
        f"目标框坐标为：{gt}\n"
        f"干扰物框列表为：{candidates}\n\n"
        "请你观察该图像，从**空间位置关系**的角度，描述该目标相对于图像中其它物体或背景的独特位置特征。\n"
        "⚠️ 输出要求：请仅输出一句中文描述，突出该目标在空间布局上的显著位置特征，用于帮助他人快速识别图中待跟踪目标。"
    )

def build_prompt_level3_cn(gt: dict, candidates: List[dict], category: str = None) -> str:
    category_str = f"该目标属于“{category}”类别，" if category else ""
    return (
        f"你正在查看一幅图像，其中的待跟踪目标物体已用绿色边界框标出，{category_str}干扰物体则以红色框标出。\n\n"
        f"目标框坐标为：{gt}\n"
        f"干扰物框列表为：{candidates}\n\n"
        "图像中干扰物体与目标在外观特征上具有较高相似性，容易引起混淆。\n"
        "请你仔细分析待跟踪目标与图中其它干扰物体之间可能导致误识的细节相似点，"
        "并明确给出最具代表性的区别特征，可有效区分目标与干扰物。\n\n"
        "⚠️ 输出要求：仅输出一句中文描述，聚焦目标最具辨识度的关键特征，"
        "该特征应能区分目标与其它干扰物体。"
    )


def build_prompt_multi_cn(gt: dict, candidates: List[dict], category: str = None) -> Dict[str, str]:
    return {
        "level_1": build_prompt_level1_cn(gt, candidates, category),
        "level_2": build_prompt_level2_cn(gt, candidates, category),
        "level_3": build_prompt_level3_cn(gt, candidates, category),
    }



def build_structured_prompt(gt: dict, candidates: List[dict], category: str = None) -> Dict[str, str]:
    """
    构建认知语言学引导下的统一描述生成 Prompt（中英文）
    输出结构固定为：
    【位置特征】，【外观特征】的【主体】在【动作状态】，【相似物体指引】。

    Args:
        gt: dict, 目标框坐标
        candidates: list[dict], 干扰框坐标列表
        category: str, 类别名称（如 "cat"）

    Returns:
        dict: {
            "zh": 中文 prompt,
            "en": English prompt
        }
    """
    category_zh = f"其类别为「{category}」" if category else ""
    category_en = f"with category \"{category}\"" if category else ""

    zh_prompt = (
        f"你正在查看一幅图像，其中包含一个明确标注的待跟踪目标（已用绿色边界框框出），{category_zh}。\n"
        "图像中还存在多个视觉上相似的干扰物（用红色框标出），可能会对目标识别造成干扰。\n\n"
        f"目标框坐标为：{gt}\n干扰物框坐标列表为：{candidates}\n\n"
        "请你从图像内容中提取有效信息，并结合认知语言学中的**具象化（concretization）**与**显著性引导（saliency guiding）**原则，从不同角度对该目标进行多层次结构化描述。用于帮助他人快速识别图中待跟踪目标。\n\n"
        "输出请严格按照以下 JSON 格式返回：\n\n"
        "```\n"
        "{\n"
        "  \"level1\": \"位置特征\",\n"
        "  \"level2\": \"外观特征\",\n"
        "  \"level3\": \"动作状态\",\n"
        "  \"level4\": \"相似物体指引\"\n"
        "}\n"
        "```\n\n"
        "说明：\n"
        "- 位置特征：绝对性描述，如“草地中央”、“树林边缘”、“道路中间”，避免使用图像左上角等坐标信息；\n"
        "- 外观特征：目标的静态描述，尽可能全面且有辨识性；\n"
        "- 动作状态：目标的动态描述，或其承载物关系；\n"
        "- 相似物体指引：目标与干扰物的差异化描述，若无显著差异，请基于位置关系排除（如“右侧有一个黑色汽车，车尾悬挂着黑色的车牌”）。\n\n"
        "⚠️ 请直接输出，无需解释过程，也不要使用图像坐标或技术性术语。"
    )

    # en_prompt = (
    #     f"You are observing an image containing a target object marked clearly by a green bounding box. The target is a {category_en}.\n"
    #     "There are also several visually similar distractor objects marked with red bounding boxes, which may interfere with identifying the correct target.\n\n"
    #     f"Target box coordinates: {gt}\nDistractor box coordinates list: {candidates}\n\n"
    #     "Extract critical visual information from the image and provide a concise, structured, multi-level description of the target object. "
    #     "Use principles of cognitive linguistics, specifically **concretization** and **saliency guiding**, to help quickly and accurately identify the target.\n\n"
    #     "Your response must strictly follow this JSON format:\n\n"
    #     "```\n"
    #     "{\n"
    #     "  \"level1\": \"Location Feature\",\n"
    #     "  \"level2\": \"Appearance Feature\",\n"
    #     "  \"level3\": \"Action State\",\n"
    #     "  \"level4\": \"Distractor Differentiation\"\n"
    #     "}\n"
    #     "```\n\n"
    #     "Instructions:\n"
    #     "- **Location Feature**: Describe the absolute location or surrounding context of the target, e.g., \"center of the grass area\", \"edge of the forest\", or \"middle of the road\". Avoid relative terms like 'top-left corner'.\n"
    #     "- **Appearance Feature**: Clearly describe the static visual attributes of the target with distinctive details.\n"
    #     "- **Action State**: Describe the current dynamic action, posture, or interaction with any objects.\n"
    #     "- **Distractor Differentiation**: Clearly indicate visual or positional differences between the target and similar distractors. If no significant visual difference exists, differentiate using positional context (e.g., \"to the right of a black car with a visible black license plate\").\n\n"
    #     "⚠️ Only provide the requested JSON response. Avoid including explanations, coordinates, or technical jargon."
    # )

    en_prompt = f"""
You are observing an image containing a target object marked clearly by a green bounding box. The target is a {category_en}.
There are also several visually similar distractor objects marked with red bounding boxes, which may interfere with identifying the correct target.

Target box coordinates: {gt}
Distractor box coordinates list: {candidates}

Your task is to produce a concise, structured, multi-level semantic description of the tracking target, guided strictly by two principles from cognitive linguistics: concretization (vivid, specific, and easily imaginable details) and saliency guiding (highlighting distinctive features that rapidly differentiate the target from distractors).

⚠️  All green or red bounding boxes are provided only to help you analyze the scene. **Do NOT mention any bounding boxes, coordinates, or technical annotation terms in your description.**

Return your answer strictly in this JSON format:
{{ "level1": "Location Feature", "level2": "Appearance Description", "level3": "Dynamic State Description", "level4": "Distractor Differentiation" }}

Instructions for Description Generation
------------
- **Location Feature**  
  • Begin with a preposition and end with a comma.  
  • Describe the semantic location (e.g., “At the center of the roadway,”).  
  • Never include coordinates or box references.

- **Appearance Description** 
  • Depending on the scenario, use one of these generalized description formats:
  1️⃣ Standalone target → “a/an [adjective(s)] [object]”  
  2️⃣ Active relation  → “a/an [adjective(s)] [object] on/in [carrier]”  
  3️⃣ *Passive relation* → Use any appropriate passive wording to indicate a carrier, e.g., “a/an [adjective(s)] [object] held by [carrier]”, “a/an [adjective(s)] [object] carried by [carrier]”, or similar.  
  • Always include **color + object type** of the target, supplemented with the most salient visual attributes (shape, size, texture, distinctive features). 
  • If a carrier is mentioned, provide a generalized but clear attribute of the carrier (e.g., color, type, or relevant characteristics).

- **Dynamic State Description**  
  • Output a **complete verb phrase** that seamlessly continues the sentence.  
  • Describe the target’s motion **as accurately and specifically as possible** (e.g., “is accelerating toward the intersection”, “is gently swaying in the wind”).  
  • If the target is on / attached to a carrier, describe the carrier’s motion (e.g., “the silver SUV is moving slowly.”). Otherwise, start with the verb for the target itself (e.g., “is hovering in place”, “is stationary”).  

  - **Distractor Differentiation**  
  • Describe **all easily confused distractors**, always using the target as the reference point. Prefer clear directional or positional terms (left / right / front / behind / above / below). 
  • To avoid confusion with the target description, each distractor description must begin with “to the target’s [direction]” (e.g., to the target’s left, behind the target), clearly separating distractors from the main object.
  • **First try a single, summarizing sentence** that links multiple distractors with parallel phrases; if the content is too dense, split into two or three short sentences, each covering one or two distractors.   
  • Each mention must pair the distractor’s **position** with a **salient appearance or action cue**. 
  • Avoid vague comparative wording such as “different from” or “unlike”; always specify exactly **where** each distractor is relative to the target.
  • **If a red-boxed region carries no clear semantic content (e.g., pure background), you may ignore that red box. If no suitable red boxes remain, identify potential distractors based on scene and target semantics and provide a differentiating description accordingly.**

Formatting rules
----------------
• Output **only** the JSON object—no markdown fences, no additional text.  
• Preserve the exact key names and comma placements shown above.
"""
    
# - **Distractor Differentiation**  
#   • Distinguish the target by stating a **clear directional or positional relationship** (left / right / front / behind / above / below) .  
#   • Use concise expressions such as:  
#     – “to the right of a black SUV in the adjacent lane,”  
#     – “directly below the overhead traffic lights,”  
#     – “in front of a stopped yellow taxi,”  
#     – “behind a red pickup truck,” etc.  
#   • Avoid vague comparative phrases like “different from” or “unlike”; always specify **where** the distractor is relative to the target.

# • Carefully examine potential distractors and judge whether they carry clear semantic content:
# If distractors have clear semantic meaning, provide generalized descriptions clearly differentiating their location, appearance, or motion relative to the tracking target. First try a single, summarizing sentence** that links multiple distractors with parallel phrases; if the content is too dense, split into two or three short sentences, each covering one or two distractors.
# If distractors lack clear semantic content (e.g., purely background areas), proactively infer plausible distractors from scene semantics and clearly, generally, describe their differences compared to the tracking target (e.g., “Although similarly colored backgrounds surround the target, these areas lack the distinct structural contours and features of the target.”).
# • Describe **all easily confused distractors**, always using the target as the reference point. Prefer clear directional or positional terms (left / right / front / behind / above / below). 
# • Avoid vague comparative wording such as “different from” or “unlike”; always specify exactly **where** each distractor is relative to the target.
  
#   • Describe **all easily confused distractors**, always using the target as the reference point. Prefer clear directional or positional terms (left / right / front / behind / above / below). 
#   • **First try a single, summarizing sentence** that links multiple distractors with parallel phrases; if the content is too dense, split into two or three short sentences, each covering one or two distractors.   
#   • Each mention must pair the distractor’s **position** with a **salient appearance or action cue**. 
#   • Avoid vague comparative wording such as “different from” or “unlike”.
#   • **If a red-boxed region carries no clear semantic content, you may ignore that red box. If no suitable red boxes remain, identify potential distractors based on scene and target semantics and provide a distractors description accordingly.**
#   • Never include analytical explanations, assumptions about viewer confusion, or subjective judgments about why something might be mistaken for the target.

#  (e.g., avoid “could be mistaken,” “looks like,” etc.)
    


#   • Ignore red boxes without semantic content. If none remain, infer plausible distractors based on the scene and describe them.
#   • For each distractor, provide a concise description of its position and salient visual or motion features. 
#   • Use clear spatial terms (e.g., left, right, behind, above) and short factual phrases, strictly using the tracking target as the reference point (e.g., to its left, behind it).
#   • Prefer a single summarizing sentence linking multiple distractors. If needed, use two or three short, parallel sentences.
  

# - **Distractor Differentiation**  
#   • Describe each distractor using a fixed sentence structure:
#     to target's [direction], a/some [adjective + salient feature] [distractor category] [optional motion/state].
#   • Use clear spatial terms such as left, right, behind, above, etc.
#   • Descriptions must be purely visual and objective, with no reasoning, comparisons, or interpretations.
#   • Do not include phrases like “might be mistaken,” “looks like,” “similar to,” etc.
#   • If no meaningful distractors exist within red boxes, identify plausible ones based on the scene and describe them using the same template.
#   • All distractor descriptions must strictly follow this template without deviation.

    return {
        "zh": zh_prompt,
        "en": en_prompt
    }

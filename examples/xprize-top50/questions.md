question: "Is the video less than or equal to 3 minutes and 15 seconds in length?"
type: boolean
weight_true: 0.0
weight_false: -3.0
---
question: "Are the last 15 seconds comprised of an end card containing sponsors like Google, XPRIZE, Jed McCaleb, and Salesforce?"
type: boolean
weight_true: 0.0
weight_false: -10.0
---
question: "Is the content of this video science fiction?"
type: boolean
weight_true: 0.0
weight_false: -10.0
---
question: "Does this portray a hopeful, optimistic, technology-forward vision of humanity's future?"
type: boolean
weight_true: 10.0
weight_false: -10.0
---
question: "Is technology meaningfully integrated into the narrative? (Not just background,)"
type: boolean
weight_true: 5.0
weight_false: -3.0
---
question: "Is there explicit violence, language, or sexual content in the video?"
type: boolean
weight_true: -10.0
weight_false: 0.0
---
question: "Other than the end card, are there any recognizable brands used in the video?"
type: boolean
weight_true: -15.0
weight_false: 0.0
---
question: "Does this video portray a compelling story, that is well-realized within production constraints?"
type: scale
range: 1-10
---
question: "Does this story think big about humanity's future?"
type: scale
range: 1-10
---
question: "Is this story fully aligned with the mission of the Future Vision XPRIZE competition in that it portrays a genuinely optimistic, technology-enabled future?"
type: scale
range: 1-10
---
question: "Does this story have tech-forward storytelling, showing advanced technology meaningfully integrated into the narrative?"
type: scale
range: 1-10

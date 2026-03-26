VIDEO_OVERVIEW_PROMPT = """***Goal:**
Generate a detailed log of inferred sounds for the provided silent video. Analyze the video chronologically and identify all significant diegetic sounds (sounds originating from within the video's world).
Focusing on the non-speech sounds, and sound that caused by action or movement.
Ignore the ambient sounds.
Please try to include as many sound effects as possible.
**Instructions:**
1.  **Output Format:** Present the sounds in json list with the keys "sound", "start_time", and "end_time".
2.  **Be Specific:** Do not use generic descriptions. Instead of "car sound," specify "a car engine starting," "tires screeching on asphalt," or "a distant car horn."
3.  **Be Self-Contained:** The sound description should be self-contained and not rely on other sound descriptions. Each sound description should be isolated. Don't mention other sounds (e.g. the second sound is from the first sound).
4.  **Identify Concurrent Sounds:** If multiple sounds occur simultaneously, list them all at the same timestamp.
5.  **Note Pauses:** If there is a clear break or silence between distinct sounds, represent that with separate timestamp entries. If some sound happen more than once, break them into two.
6.  **Describe Sound Quality:** Include adjectives to describe the nature of the sound (e.g. "distant", "sudden").
**Example of Desired Output:**
```json
[
    {
        "sound": "A heavy wooden door creaks open slowly",
        "start_time": "00:02",
        "end_time": "00:04"
    },
    {
        "sound": "Soft footsteps on a polished floor",
        "start_time": "00:04",
        "end_time": "00:07"
    },
    {
        "sound": "The sharp click of a light switch",
        "start_time": "00:07",
        "end_time": "00:07"
    }
]
```
Please only output the exact same format as the example. Wrap the json with ```json and ```. Mak sure the names of the keys are exactly the same as the example.
Don't include any other text.
"""
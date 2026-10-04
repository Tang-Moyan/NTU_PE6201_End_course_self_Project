"""Generate tests/cases.json from the knowledge base. Run from repo root."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.retrieve import _candidate_score, load_quotes, normalize, retrieve

OUT = Path(__file__).with_name("cases.json")

HP_IDX = [0, 1, 2, 3, 4, 5, 8, 7]  # skip index 6 (hard to paraphrase naturally under fuzzy rules)
RM_IDX = list(range(8))
LOTR_IDX = list(range(8))


def by_source(entries):
    hp, rm, lotr = [], [], []
    for e in entries:
        if e.source == "Harry Potter":
            hp.append(e)
        elif e.source == "Rick And Morty":
            rm.append(e)
        else:
            lotr.append(e)
    return hp, rm, lotr


def pick_core24(entries):
    hp, rm, lotr = by_source(entries)
    return [hp[i] for i in HP_IDX] + [rm[i] for i in RM_IDX] + [lotr[i] for i in LOTR_IDX]


VARIATION_WRAPPERS = [
    "Someone in chat paraphrased it perfectly: {v}",
    "I can't quote it exactly, but the vibe was \"{v}\"",
    "The way they put it stuck with me — {v}",
    "Not word-for-word, more like: {v}",
]

EMBEDDED_WRAPPERS = [
    "The message ended with this exact line from the film: {q}",
    "I pasted the subtitle because it nailed the moment: {q}",
    "Still stuck on what they said verbatim: {q}",
    "Group chat devolved into quoting, and someone dropped: {q}",
    "Hard to explain without quoting directly: {q}",
    "After the argument cooled, one line kept replaying: {q}",
]


def variation_message(variation: str, slot: int) -> str:
    template = VARIATION_WRAPPERS[slot % len(VARIATION_WRAPPERS)]
    return template.format(v=variation)


def embedded_message(quote: str, slot: int) -> str:
    template = EMBEDDED_WRAPPERS[slot % len(EMBEDDED_WRAPPERS)]
    return template.format(q=quote)


# Natural paraphrases: fuzzy-match canonical quote; not equal to quote or any variation.
PARAPHRASES: dict[str, str] = {
    "It takes a great deal of bravery to stand up your enemies but a great deal more to stand up to your friends.": (
        "It takes a great deal of bravery to stand up your enemies, but even more courage to stand up to your friends."
    ),
    "It is not our abilities that show what we truly are. It is our choices.": (
        "It is not our abilities that show what we truly are; it is the choices we make."
    ),
    "No spell can reawaken the dead, Harry.": (
        "No spell could reawaken the dead, Harry."
    ),
    "Forgive my mawkishness, Harry. I'm an old man.": (
        "Forgive my mawkishness, Harry — I am an old man, and I get sentimental."
    ),
    "Once again, I must ask too much of you, Harry.": (
        "Once again, I must ask far too much of you, Harry."
    ),
    "He'll never be gone. Not as long as those who remain are loyal to him.": (
        "He'll never truly be gone, not as long as those who remain are loyal to him."
    ),
    "We're gonna make them proud, Neville. That's a promise.": (
        "We will make them proud, Neville — that's a promise."
    ),
    "Working hard is important, but there's something that matters even more: believing in yourself.": (
        "Working hard is important, but there is something that matters even more: trusting yourself."
    ),
    "Morty, You're right. Town is bad. But that doesn't mean that we have to be.": (
        "Morty, you're right that town is bad, but that doesn't mean we have to be bad too."
    ),
    "But speeches are for campaigning. Now, is the time for action.": (
        "Speeches are for campaigning — now is the time for action."
    ),
    "I thought the whole point of having a dog was to feel superior, Jerry?": (
        "Jerry, I thought the whole point of having a dog was feeling superior?"
    ),
    "The universe is a crazy, chaotic place.": (
        "The universe remains a crazy, chaotic place."
    ),
    "Parents are just kids having kids.": (
        "Parents are mostly just kids having kids."
    ),
    "Our people will get more from the idea he represented than from the Jelly Bean he actually was.": (
        "Our people will get more from the idea he stood for than from the Jelly Bean he actually was."
    ),
    "Nobody exists on purpose, nobody belongs anywhere, everybody's gonna die, come watch TV.": (
        "Nobody exists on purpose, nobody belongs anywhere, everybody's gonna die — so come watch TV."
    ),
    "Scientifically, traditions are an idiot thing.": (
        "Scientifically, traditions are a pretty idiot thing."
    ),
    "Hobbits really are amazing creatures. You can learn all that there is to know about their ways in a month and yet, after a hundred years, they can still surprise you.": (
        "Hobbits really are amazing creatures — you can learn their ways in a month, yet after a hundred years they can still surprise you."
    ),
    "I have nothing greater to give than the gift you already bear.": (
        "I have nothing greater to give you than the gift you already bear."
    ),
    "A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to.": (
        "A wizard is never late, Frodo Baggins. Nor is he early, and he arrives precisely when he means to."
    ),
    "All we have to decide is what to do with the time that is given to us.": (
        "All we have to decide is what we do with the time that is given to us."
    ),
    "One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them.": (
        "One Ring to rule them all, One Ring to find them, One Ring to bring them all and in darkness bind them."
    ),
    "Even the smallest person can change the course of the future.": (
        "Even the smallest person can change the course of what comes next."
    ),
    "You shall not pass.": (
        "You shall not go past me here."
    ),
    "It is a strange fate that we should suffer so much fear and doubt over so small a thing.": (
        "It is a strange fate that we should suffer so much fear and doubt over such a small thing."
    ),
}

NORMALIZED_FORMS: dict[str, str] = {
    "It takes a great deal of bravery to stand up your enemies but a great deal more to stand up to your friends.": (
        "IT TAKES A GREAT DEAL OF BRAVERY TO STAND UP YOUR ENEMIES BUT A GREAT DEAL MORE TO STAND UP TO YOUR FRIENDS."
    ),
    "It is not our abilities that show what we truly are. It is our choices.": (
        "it is not our abilities that show what we truly are. it is our choices."
    ),
    "No spell can reawaken the dead, Harry.": (
        "No spell can reawaken the dead, Harry!!!"
    ),
    "Forgive my mawkishness, Harry. I'm an old man.": (
        "Forgive my mawkishness, Harry. I’m an old man."
    ),
    "Once again, I must ask too much of you, Harry.": (
        "once again, i must ask too much of you, harry."
    ),
    "He'll never be gone. Not as long as those who remain are loyal to him.": (
        "He'll never be gone — not as long as those who remain are loyal to him."
    ),
    "We're gonna make them proud, Neville. That's a promise.": (
        "WE'RE GONNA MAKE THEM PROUD, NEVILLE. THAT'S A PROMISE."
    ),
    "Working hard is important, but there's something that matters even more: believing in yourself.": (
        "Working hard is important, but there's something that matters even more: believing in yourself!!!"
    ),
    "Morty, You're right. Town is bad. But that doesn't mean that we have to be.": (
        "MORTY, YOU'RE RIGHT. TOWN IS BAD. BUT THAT DOESN'T MEAN THAT WE HAVE TO BE."
    ),
    "But speeches are for campaigning. Now, is the time for action.": (
        "But speeches are for campaigning. Now, is the time for action!!!"
    ),
    "I thought the whole point of having a dog was to feel superior, Jerry?": (
        "i thought the whole point of having a dog was to feel superior, jerry?"
    ),
    "The universe is a crazy, chaotic place.": (
        "The universe is a crazy, chaotic place!!!"
    ),
    "Parents are just kids having kids.": (
        "PARENTS ARE JUST KIDS HAVING KIDS."
    ),
    "Our people will get more from the idea he represented than from the Jelly Bean he actually was.": (
        "Our people will get more from the idea he represented than from the Jelly Bean he actually was!!!"
    ),
    "Nobody exists on purpose, nobody belongs anywhere, everybody's gonna die, come watch TV.": (
        "nobody exists on purpose, nobody belongs anywhere, everybody's gonna die, come watch TV."
    ),
    "Scientifically, traditions are an idiot thing.": (
        "SCIENTIFICALLY, TRADITIONS ARE AN IDIOT THING."
    ),
    "Hobbits really are amazing creatures. You can learn all that there is to know about their ways in a month and yet, after a hundred years, they can still surprise you.": (
        "hobbits really are amazing creatures. you can learn all that there is to know about their ways in a month and yet, after a hundred years, they can still surprise you."
    ),
    "I have nothing greater to give than the gift you already bear.": (
        "I have nothing greater to give than the gift you already bear!!!"
    ),
    "A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to.": (
        "A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to!!!"
    ),
    "All we have to decide is what to do with the time that is given to us.": (
        "ALL WE HAVE TO DECIDE IS WHAT TO DO WITH THE TIME THAT IS GIVEN TO US."
    ),
    "One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them.": (
        "ONE RING TO RULE THEM ALL, ONE RING TO FIND THEM, ONE RING TO BRING THEM ALL, AND IN THE DARKNESS BIND THEM."
    ),
    "Even the smallest person can change the course of the future.": (
        "even the smallest person can change the course of the future."
    ),
    "You shall not pass.": (
        "YOU SHALL NOT PASS."
    ),
    "It is a strange fate that we should suffer so much fear and doubt over so small a thing.": (
        "It is a strange fate that we should suffer so much fear and doubt over so small a thing!!!"
    ),
}

# (full message text, quote_a, quote_b) — same source, coherent chat context.
MULTI_MESSAGES: list[tuple[str, str, str]] = [
    (
        "Our charity run group chat turned into a Potter pep talk: We're gonna make them proud, Neville. That's a promise. "
        "Then someone replied Maybe you don't have to do this by yourself, mate.",
        "We're gonna make them proud, Neville. That's a promise.",
        "Maybe you don't have to do this by yourself, mate.",
    ),
    (
        "We were arguing about calling out a toxic club leader when two lines got pasted back-to-back — "
        "It takes a great deal of bravery to stand up your enemies but a great deal more to stand up to your friends. "
        "and It is not our abilities that show what we truly are. It is our choices.",
        "It takes a great deal of bravery to stand up your enemies but a great deal more to stand up to your friends.",
        "It is not our abilities that show what we truly are. It is our choices.",
    ),
    (
        "After a sad rewatch someone wrote No spell can reawaken the dead, Harry. "
        "Another person answered with Now, fear makes people do terrible things, Harry.",
        "No spell can reawaken the dead, Harry.",
        "Now, fear makes people do terrible things, Harry.",
    ),
    (
        "Welcome-back thread for a teammate who keeps the group together: There's no Hogwarts without you, Hagrid. "
        "Someone added Working hard is important, but there's something that matters even more: believing in yourself.",
        "There's no Hogwarts without you, Hagrid.",
        "Working hard is important, but there's something that matters even more: believing in yourself.",
    ),
    (
        "Mission-planning voice note quoted two lines about burden: Once again, I must ask too much of you, Harry. "
        "and Harry, it's you that has to go on. Not me. Not Hermione. You.",
        "Once again, I must ask too much of you, Harry.",
        "Harry, it's you that has to go on. Not me. Not Hermione. You.",
    ),
    (
        "Memorial post for a mentor who left the company: He'll never be gone. Not as long as those who remain are loyal to him. "
        "A comment underneath said I trusted the man I know.",
        "He'll never be gone. Not as long as those who remain are loyal to him.",
        "I trusted the man I know.",
    ),
    (
        "Exam-week support thread mixed encouragement with priorities: Working hard is important, but there's something that matters even more: believing in yourself. "
        "Later the same person posted There are more important things. Friendship and bravery.",
        "Working hard is important, but there's something that matters even more: believing in yourself.",
        "There are more important things. Friendship and bravery.",
    ),
    (
        "Snarky study break: Believe me, professor, I've been asking myself the same question for six years. "
        "followed immediately by Clearly, fame isn't everything.",
        "Believe me, professor, I've been asking myself the same question for six years.",
        "Clearly, fame isn't everything.",
    ),
    (
        "Rick and Morty night in the dorm: Homework is stupid, the whole point is to get less of it. "
        "My roommate countered with That's planning for failure, Morty, even dumber than regular planning.",
        "Homework is stupid, the whole point is to get less of it.",
        "That's planning for failure, Morty, even dumber than regular planning.",
    ),
    (
        "Debate club used two clips about talk versus action: Morty, You're right. Town is bad. But that doesn't mean that we have to be. "
        "Then someone dropped But speeches are for campaigning. Now, is the time for action.",
        "Morty, You're right. Town is bad. But that doesn't mean that we have to be.",
        "But speeches are for campaigning. Now, is the time for action.",
    ),
    (
        "Family group chat after a rough week: Parents are just kids having kids. "
        "My cousin replied Nobody exists on purpose, nobody belongs anywhere, everybody's gonna die, come watch TV.",
        "Parents are just kids having kids.",
        "Nobody exists on purpose, nobody belongs anywhere, everybody's gonna die, come watch TV.",
    ),
    (
        "Science fair drama recap: Scientifically, traditions are an idiot thing. "
        "and You can't make people like you. You just have to wait for hating you to bore them.",
        "Scientifically, traditions are an idiot thing.",
        "You can't make people like you. You just have to wait for hating you to bore them.",
    ),
    (
        "Town-hall thread about a disgraced leader: Our people will get more from the idea he represented than from the Jelly Bean he actually was. "
        "Someone else pasted Because when you're an asshole, it doesn't matter how right you are, nobody wants to give you the satisfaction.",
        "Our people will get more from the idea he represented than from the Jelly Bean he actually was.",
        "Because when you're an asshole, it doesn't matter how right you are, nobody wants to give you the satisfaction.",
    ),
    (
        "Road-trip rant after a mess: The universe is a crazy, chaotic place. "
        "Then the next message was Otherwise, you are just an inert chunk of randomly assembled molecules, drifting wherever the universe blows you.",
        "The universe is a crazy, chaotic place.",
        "Otherwise, you are just an inert chunk of randomly assembled molecules, drifting wherever the universe blows you.",
    ),
    (
        "Pet debate in the apartment chat: I thought the whole point of having a dog was to feel superior, Jerry? "
        "which spiraled into To live is to risk it all.",
        "I thought the whole point of having a dog was to feel superior, Jerry?",
        "To live is to risk it all.",
    ),
    (
        "Anti-bureaucracy rant night: You want to be safe from the government, so you became a stupid government. That makes every Rick here less Rick than me. "
        "When you know nothing matters, the universe is yours. came up right after.",
        "You want to be safe from the government, so you became a stupid government. That makes every Rick here less Rick than me.",
        "When you know nothing matters, the universe is yours.",
    ),
    (
        "School obligations vs adventures thread: How is my son supposed to pass his classes if you keep dragging him off for a high concept Sci Fi rigmarole? "
        "Someone answered with I'm an idiot, and I love my son.",
        "How is my son supposed to pass his classes if you keep dragging him off for a high concept Sci Fi rigmarole?",
        "I'm an idiot, and I love my son.",
    ),
    (
        "Late-night existential voice notes: When you know nothing matters, the universe is yours. "
        "then Weddings are basically funerals with cake.",
        "When you know nothing matters, the universe is yours.",
        "Weddings are basically funerals with cake.",
    ),
    (
        "LOTR rewatch group: That there's some good in this world, Mr. Frodo. And it's worth fighting for. "
        "Sam's next line in the thread was I can't carry it for you, but I can carry you.",
        "That there's some good in this world, Mr. Frodo. And it's worth fighting for.",
        "I can't carry it for you, but I can carry you.",
    ),
    (
        "Travel delays group chat: A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to. "
        "Someone answered All we have to decide is what to do with the time that is given to us.",
        "A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to.",
        "All we have to decide is what to do with the time that is given to us.",
    ),
    (
        "Hiking safety argument: You shall not pass. "
        "My friend replied with Whatever happens, stay with me. I'll look out for you.",
        "You shall not pass.",
        "Whatever happens, stay with me. I'll look out for you.",
    ),
    (
        "Pre-game hype in our fantasy league chat: A day may come when the courage of men fails, when we forsake our friends and break all bonds of fellowship, but it is not this day. "
        "Then Long ago we fought and died together. We've come to honor that allegiance.",
        "A day may come when the courage of men fails, when we forsake our friends and break all bonds of fellowship, but it is not this day.",
        "Long ago we fought and died together. We've come to honor that allegiance.",
    ),
    (
        "Gift exchange got emotional: I have nothing greater to give than the gift you already bear. "
        "which made someone post Hobbits really are amazing creatures. You can learn all that there is to know about their ways in a month and yet, after a hundred years, they can still surprise you.",
        "I have nothing greater to give than the gift you already bear.",
        "Hobbits really are amazing creatures. You can learn all that there is to know about their ways in a month and yet, after a hundred years, they can still surprise you.",
    ),
    (
        "Ring lore meme thread turned serious: One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them. "
        "Then It is a strange fate that we should suffer so much fear and doubt over so small a thing.",
        "One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them.",
        "It is a strange fate that we should suffer so much fear and doubt over so small a thing.",
    ),
]

CROSS_MESSAGES: list[tuple[str, str, str]] = [
    (
        "Two fandom lines I pair when talking about agency: It is not our abilities that show what we truly are. It is our choices. "
        "and All we have to decide is what to do with the time that is given to us.",
        "It is not our abilities that show what we truly are. It is our choices.",
        "All we have to decide is what to do with the time that is given to us.",
    ),
    (
        "When morale crashed, one friend sent There are more important things. Friendship and bravery. "
        "and another replied Nobody exists on purpose, nobody belongs anywhere, everybody's gonna die, come watch TV.",
        "There are more important things. Friendship and bravery.",
        "Nobody exists on purpose, nobody belongs anywhere, everybody's gonna die, come watch TV.",
    ),
    (
        "Comparing toxic-system speeches we have saved: Morty, You're right. Town is bad. But that doesn't mean that we have to be. "
        "next to He'll never be gone. Not as long as those who remain are loyal to him.",
        "Morty, You're right. Town is bad. But that doesn't mean that we have to be.",
        "He'll never be gone. Not as long as those who remain are loyal to him.",
    ),
    (
        "Boundary-setting memes in one message: You shall not pass. "
        "and I thought the whole point of having a dog was to feel superior, Jerry?",
        "You shall not pass.",
        "I thought the whole point of having a dog was to feel superior, Jerry?",
    ),
    (
        "Anti-establishment rants we juxtapose: You want to be safe from the government, so you became a stupid government. That makes every Rick here less Rick than me. "
        "alongside Once again, I must ask too much of you, Harry.",
        "You want to be safe from the government, so you became a stupid government. That makes every Rick here less Rick than me.",
        "Once again, I must ask too much of you, Harry.",
    ),
    (
        "Comfort lines we swap when friends burn out: That there's some good in this world, Mr. Frodo. And it's worth fighting for. "
        "and Parents are just kids having kids.",
        "That there's some good in this world, Mr. Frodo. And it's worth fighting for.",
        "Parents are just kids having kids.",
    ),
    (
        "Study motivation vs Rick logic: Working hard is important, but there's something that matters even more: believing in yourself. "
        "Homework is stupid, the whole point is to get less of it.",
        "Working hard is important, but there's something that matters even more: believing in yourself.",
        "Homework is stupid, the whole point is to get less of it.",
    ),
    (
        "Chaos travel stories in one rant: A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to. "
        "The universe is a crazy, chaotic place.",
        "A wizard is never late, Frodo Baggins. Nor is he early; he arrives precisely when he means to.",
        "The universe is a crazy, chaotic place.",
    ),
    (
        "Promise lines we quote before tough team projects: We're gonna make them proud, Neville. That's a promise. "
        "Long ago we fought and died together. We've come to honor that allegiance.",
        "We're gonna make them proud, Neville. That's a promise.",
        "Long ago we fought and died together. We've come to honor that allegiance.",
    ),
    (
        "Risk-taking pep talk mashup: To live is to risk it all. "
        "Even the smallest person can change the course of the future.",
        "To live is to risk it all.",
        "Even the smallest person can change the course of the future.",
    ),
    (
        "Trust after messy revelations: I trusted the man I know. "
        "I have nothing greater to give than the gift you already bear.",
        "I trusted the man I know.",
        "I have nothing greater to give than the gift you already bear.",
    ),
    (
        "Burnout versus political cynicism: Crossed your brilliant mind that I don't want to do this anymore? "
        "Our people will get more from the idea he represented than from the Jelly Bean he actually was.",
        "Crossed your brilliant mind that I don't want to do this anymore?",
        "Our people will get more from the idea he represented than from the Jelly Bean he actually was.",
    ),
    (
        "Surprise and tradition argument: Hobbits really are amazing creatures. You can learn all that there is to know about their ways in a month and yet, after a hundred years, they can still surprise you. "
        "Scientifically, traditions are an idiot thing.",
        "Hobbits really are amazing creatures. You can learn all that there is to know about their ways in a month and yet, after a hundred years, they can still surprise you.",
        "Scientifically, traditions are an idiot thing.",
    ),
    (
        "Support versus being right online: I can't carry it for you, but I can carry you. "
        "Because when you're an asshole, it doesn't matter how right you are, nobody wants to give you the satisfaction.",
        "I can't carry it for you, but I can carry you.",
        "Because when you're an asshole, it doesn't matter how right you are, nobody wants to give you the satisfaction.",
    ),
    (
        "Power and loss in one breath: One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them. "
        "No spell can reawaken the dead, Harry.",
        "One Ring to rule them all, One Ring to find them, One Ring to bring them all, and in the darkness bind them.",
        "No spell can reawaken the dead, Harry.",
    ),
    (
        "Fear-driven politics versus ruthless action: Now, fear makes people do terrible things, Harry. "
        "But speeches are for campaigning. Now, is the time for action.",
        "Now, fear makes people do terrible things, Harry.",
        "But speeches are for campaigning. Now, is the time for action.",
    ),
    (
        "Belonging in a group chat: There's no Hogwarts without you, Hagrid. "
        "Even the smallest person can change the course of the future.",
        "There's no Hogwarts without you, Hagrid.",
        "Even the smallest person can change the course of the future.",
    ),
    (
        "Oddly specific pep talk combo: Inside every girl, a secret swan slumbers longing to burst forth and take flight. "
        "When you know nothing matters, the universe is yours.",
        "Inside every girl, a secret swan slumbers longing to burst forth and take flight.",
        "When you know nothing matters, the universe is yours.",
    ),
    (
        "Mortality talk that got too heavy: Death is just another path. One that we all must take. "
        "No spell can reawaken the dead, Harry.",
        "Death is just another path. One that we all must take.",
        "No spell can reawaken the dead, Harry.",
    ),
    (
        "Oathbreakers and backup plans: They had no honor in life. They have none now in death. "
        "That's planning for failure, Morty, even dumber than regular planning.",
        "They had no honor in life. They have none now in death.",
        "That's planning for failure, Morty, even dumber than regular planning.",
    ),
    (
        "Tools choosing people and approval online: The wand chooses the wizard. "
        "You can't make people like you. You just have to wait for hating you to bore them.",
        "The wand chooses the wizard.",
        "You can't make people like you. You just have to wait for hating you to bore them.",
    ),
    (
        "Small objects, big dread: It is a strange fate that we should suffer so much fear and doubt over so small a thing. "
        "Otherwise, you are just an inert chunk of randomly assembled molecules, drifting wherever the universe blows you.",
        "It is a strange fate that we should suffer so much fear and doubt over so small a thing.",
        "Otherwise, you are just an inert chunk of randomly assembled molecules, drifting wherever the universe blows you.",
    ),
    (
        "Standing together in bad systems: A day may come when the courage of men fails, when we forsake our friends and break all bonds of fellowship, but it is not this day. "
        "Morty, You're right. Town is bad. But that doesn't mean that we have to be.",
        "A day may come when the courage of men fails, when we forsake our friends and break all bonds of fellowship, but it is not this day.",
        "Morty, You're right. Town is bad. But that doesn't mean that we have to be.",
    ),
    (
        "Catchphrase surprise chain: Ooooh wee! "
        "Always a total surprise.",
        "Ooooh wee!",
        "Always a total surprise.",
    ),
]

NO_HIT_TEXTS = [
    "Can you pick up milk on your way home?",
    "The bus was twenty minutes late again.",
    "I'll review your draft after lunch.",
    "今天下班一起去吃火锅吗？",
    "Meeting moved to Thursday at three.",
    "My phone battery dies by noon every day.",
    "The printer jammed on page four.",
    "Do we still need extra chairs for the workshop?",
    "I forgot my umbrella on the train.",
    "Let's split the taxi fare evenly.",
    "The pharmacy closes at eight tonight.",
    "Can you send the Zoom link again?",
    "Her cat knocked the plant off the shelf.",
    "I'm trying a new recipe for pasta tonight.",
    "Traffic on the expressway was awful.",
    "Please mute when you're not speaking.",
    "The museum ticket is valid for two days.",
    "I need to renew my library card.",
    "We should book flights before prices rise.",
    "The Wi-Fi password is on the fridge note.",
    "Coffee machine needs descaling again.",
    "Is the gym open on public holidays?",
    "I'll take the night shift if you cover Saturday.",
    "The package arrived but the box was dented.",
]

NEAR_MISS_TEXTS = [
    "I love my friends so much it hurts sometimes.",
    "I'll just watch TV and call it a night.",
    "Homework can wait until Sunday.",
    "Friendship matters more than being right online.",
    "You shall not open my snack drawer.",
    "Five more minutes and I'll leave.",
    "The universe is really, really big.",
    "Parents are exhausted this week.",
    "I trusted him with the keys once.",
    "Always a surprise when the bus shows up.",
    "I fear the dentist more than anything.",
    "There are more important things than this spreadsheet.",
    "Fame on social media fades fast anyway.",
    "Life feels like one long risk assessment lately.",
    "A wizard hat would look ridiculous on me.",
    "The ring on her finger is white gold.",
    "Death metal is too loud for studying.",
    "Stay with the group at the airport gate.",
    "Courage helps in interviews, I guess.",
    "The government form has three pages, not four.",
    "School traditions include a spring picnic.",
    "My dog likes sunbeams on the couch.",
    "Graduation speeches run too long every year.",
    "The future course catalog posted this morning.",
]


def validate_paraphrase(entry, text: str, quotes) -> None:
    q = entry.quote
    assert text != q
    nq = normalize(q)
    hay = normalize(text)
    assert nq not in hay, f"paraphrase contains full quote: {q[:40]}"
    for v in entry.variations:
        if text.strip() == v.strip() or normalize(text) == normalize(v):
            raise AssertionError(f"paraphrase equals variation: {v[:40]}")
    score = _candidate_score(nq, hay)
    assert score >= 0.86, f"paraphrase score {score} for {q[:50]}"
    matches = retrieve(text, quotes)
    found = {m.entry.quote for m in matches}
    assert q in found, f"paraphrase not retrieved: {text[:60]}"


def main() -> None:
    entries = load_quotes()
    core = pick_core24(entries)
    quote_map = {e.quote: e for e in entries}
    cases: list[dict] = []
    n = 0

    def add(result_type: str, text: str, expected: list[str]) -> None:
        nonlocal n
        n += 1
        cases.append(
            {
                "id": f"{result_type}-{n:03d}",
                "result_type": result_type,
                "text": text,
                "expected_quotes": expected,
            }
        )

    for entry in core:
        add("exact", entry.quote, [entry.quote])

    for entry in core:
        q = entry.quote
        text = NORMALIZED_FORMS[q]
        assert text != q
        assert normalize(text) == normalize(q)
        add("normalized", text, [q])

    for slot, entry in enumerate(core):
        var = next(v for v in entry.variations if normalize(v) != normalize(entry.quote))
        add("variation", variation_message(var, slot), [entry.quote])

    for entry in core:
        text = PARAPHRASES[entry.quote]
        validate_paraphrase(entry, text, tuple(entries))
        add("paraphrase", text, [entry.quote])

    for slot, entry in enumerate(core):
        add("embedded", embedded_message(entry.quote, slot), [entry.quote])

    assert len(MULTI_MESSAGES) == 24
    for text, qa, qb in MULTI_MESSAGES:
        assert quote_map[qa].source == quote_map[qb].source
        add("multi", text, [qa, qb])

    assert len(CROSS_MESSAGES) == 24
    for text, q1, q2 in CROSS_MESSAGES:
        assert quote_map[q1].source != quote_map[q2].source
        add("cross_source", text, [q1, q2])

    for text in NO_HIT_TEXTS:
        assert not retrieve(text, tuple(entries)), text
        add("no_hit", text, [])

    for text in NEAR_MISS_TEXTS:
        assert not retrieve(text, tuple(entries)), text
        add("near_miss", text, [])

    for c in cases:
        found = [m.entry.quote for m in retrieve(c["text"], tuple(entries), limit=10)]
        exp = c["expected_quotes"]
        if sorted(found) != sorted(exp):
            raise SystemExit(f"FAIL {c['id']}: found {found} expected {exp}")

    payload = {
        "result_types": {
            "exact": "User message equals the canonical quote text (character-for-character).",
            "normalized": "Only casing, punctuation, or apostrophe style differs; normalized tokens match the quote.",
            "variation": "Message uses a knowledge-base variation (often inside natural chat), not the canonical quote.",
            "paraphrase": "Natural near wording that fuzzy-matches the quote but is not the quote or a listed variation.",
            "embedded": "Full canonical quote appears naturally inside a longer chat message.",
            "multi": "One coherent message uses two quotes from the same source work.",
            "cross_source": "One coherent message uses quotes from at least two different works.",
            "no_hit": "Everyday message with no retrievable quote overlap.",
            "near_miss": "Thematic or lexical overlap without a valid quote match.",
        },
        "cases": cases,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", OUT, "total", len(cases), Counter(c["result_type"] for c in cases))


if __name__ == "__main__":
    main()

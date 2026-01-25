from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")

text = '''
"I pledge my allegiance, now and always, forever to my country," I say into my AirPhone, voice raspy from waking up so early. Much to my sleepy dismay, my bedroom window allows a few rays of light into the room, turning the comforting darkness into soft light.

Nestled in a small divot on my thick bed sheets, my ultra-thin AirPhone records my little spiel. A soft ticking noise fires from the device into the quiet safety of my room. Every word I recite must be exact.

I shift my weight, wrinkling the straightened gray covers underneath me. The sound of my rasp fails to fade away, so my pledge feels more like sandpaper in my mouth than smooth water. I clear my throat as a hologram of the national flag bursts from my Phone screen, waving powerfully. It's crazy to know that everyone across the nation starts their day the same way I do, repeating the same message, the same promise, completely in sync. Millions of people inaudible and invisible to me yet loud and clear.

I continue. "My loyalty will never falter, my obedience will never halt, and my actions will always reflect these words: timeliness for the sake of productivity; control for the sake of order."

The wide flag flourishes itself again as a systematic message says, "The President approves your pledge. Now complete your daily duties with pride."

"Now and always," I reply, and the hologram clicks off, vanishing with a swish. I slide my Phone into my hand, check my reflection in the mirror of my screen, and exit my room through automatic doors.

Like clockwork, I step into my apartment's bare kitchen, aligning my feet onto the cushy PrintPad. The Pad recognizes my footprints, then the lone refrigerated cabinet slides open. I reach for a NutriBar, and the gray doors lock themselves when they feel the weight of the Bar leave the cabinet.

I stride out of the kitchen, which hardly takes a second because of its small size. Padding across my living room, I sit on the sofa and begin to eat. I reluctantly tear open the plastic packaging of the Bar, spewing a few crumbs into my lap.

The nation issues portion-controlled NutriBars toeveryone, regardless of economic standing. But the middle class always end up with the stale ones. The rich want the best and the poor need the best, so people like my parents and I get the worst.

I take a few more bites. The taste of the Bar is neutral, but it's the only option besides water. Besides, nothing beats a full stomach of NutriBars, even if they're a little expired.

The silence of my small house pounds into my ears, intensifying the sound of my chewing. I crinkle the NutriBar wrapper to soften the quietness. My parents left for the factories an hour ago, so I'm all alone in our apartment.

I slide my AirPhone out of my pocket, the time displaying holographically when I lay the device on the sofa. Anyone between ages 14 and 17 must be on their SkyTrain by 7 o'clock sharp. No exceptions.

I've never been late for my Train before, but I've seen a boy miss his SkyTrain a few years ago. The Screeners had him bound with ElectriWrist cuffs and begging for mercy before I could even blink. Luckily, the train zoomed away just as the Screeners made the move to smash the boy's head into the ground.

That boy wasn't the only one who got screened that day. I got a partial screening for unlocking my safety belt and banging on the windows to get the Train to stop. The Screeners locked down SkyTrain 267-B and had me escorted away within seconds. My parents had to pay heavy fines for my direct violation of the national motto: timeliness for the sake of productivity; control for the sake of order. I endured a whole week of correctional therapy before the Screeners allowed me on the Trains again. At that point, no on wanted to associate with the girl who got screened.

I mentally lock the memory away and finally finish my NutriBar while pacing into the kitchen. I feed the plastic wrapping into the skinny shredder installed into the counter. Leftovers are strictly prohibited, so throwing out my food isn't ever an option. The Screeners monitor everything at every moment, even my nutrient intake in my small little apartment.

Checking the time again, I grab my dull jacket off the metal hook in the living room. I slide my arms into the familiar sleeves, pulling my hair out from underneath the collar. Then I step onto another PrintPad, and the bottom half of the window slides upwards once my footprints are recognized. After tossing on my worker's boots and sling bag, I climb out the square window and step onto the shaky fire escape. I watch the window clamp shut behind me, slowly revealing a crimson 46B-L so I know which window is mine when I come back home.

I turn away from the window to face the city and the surrounding skyline. Breathlessly smooth buildings intermix with each other, SkyRails connecting their flat roofs. But the real beauty is where the the sun looms in the distance, tinting the clouds radiant pinks, blues, and purples. I take a breath of the cool morning air, filling my entire body with a chill. The wind pricks at the back of my neck, sending light shivers up my arms and down my spine.

I take two steps towards the end of the fire escape and lean on the railing, too immune to the height to be afraid. The ant-like Screeners in their iconic red uniforms dot the streets below. They position themselves at every street corner and on every SkyRail-lined rooftop I see. I turn my gaze back to the horizon instead.

All citizens are supposed to adore the city and hail the Screeners, but after my partial screening, I turned to the distance for some sanity. I'd be screened (again) if I ever said it out loud. The Screeners know everything, and their unlimited knowledge forbids citizens from disrespecting the nation. That's why the Screeners' motto fits them perfectly: freedom within limits; security over privacy.

Skimming my fingers up and down on the fire escape railing, I stare into the distance and lose track of time. I still don't understand what I did wrong that day when the boy missed his Train. I tried to help save him from pain but ended up hurting myself and my family in the process. Either way, there's nothing I can do about it.

Shaking myself out of my thoughts, I inhale more refreshing air with a sigh. I lean my weight onto the railing before beginning to mount the steps of the metal fire escape, taking them two at a time. Even though the steps are a bit slippery from the rain last night, I manage to make it from level 46 to 60 at my usual pace. My sling bag bangs against my leg as I climb the final steps of the fire escape. I hop with a loud thump onto the roof, earning sideways glances from the Screeners.

I ignore the Screeners as I walk across the roof, ducking under the spine-like SkyRails six feet above me. The Rails create a complex metal system of transportation across every rooftop in the city. I avoid stopping at pillar 4 where all my classmates are and decide to wait for the Train on the other side of the roof instead. Groups of teens talk mindlessly, parts of conversations lost in the wind. I walk by them unseen, stopping at my usual spot.

I pull out my Phone and check the time again: 6:54. The morning wind howls in my ears as the sun peaks from the surrounding hills, warming my backside against the cool winds. I scan the roof for no one in particular, but I find myself disappointed when I see the same sixty or so teens who live in my building. They talk and laugh in their oblivious groups, leaving me to watch them. I always thought I was good at blending in, but I'm beginning to think I was sticking out the whole time.

I hopelessly toss my dirty blonde hair into a messy ponytail, small strands blowing in the wind. Suddenly I hear clanking from behind me. I turn around as Nicolette throws her sling onto the roof. She pulls herself over the ledge of the building instead of leaping like I do.

I scoot a few paces away from her, watching her with my arms crossed. Her brown eyes search for anyone,anyone, except me. I turn away from her and concentrate on the laces of my boots instead, reminding myself that I'm better off without her as my friend.

My heart beats hard against my ribs. She shouldn't have ditched me after my partial screening.

"Hey, guys!" Nicolette yells across the roof. My former group of friends greets her happily. "Wait for me!"

As the words escape her lips, a newer, stronger burst of wind fills my ears. I tilt my head upwards to watch the gray SkyTrain halts with precision above the middle of the roof. As soon as Sky Train 267-B stops, the pillars supporting the Rails shift to form ladder rungs.

Everyone on the roof begins to climb the rungs, and Nicolette and I hurry over to pillar 4, the ladder for 17-year-old's. After waiting a few seconds in line, I climb the ladder and follow the rungs through a circular opening in the SkyTrain's floor. I step off the ladder onto the bleached floor, shuffling out of the way so Nicolette can dismount from the ladder, too.

We pull out our AirPhones and hold them face-down over the Tracer, a podium-like data collector. An aqua blue striplight scans the faces of our Phones, casting blue shadows onto the white ceiling of the train. The strip of blue light shuts off and our names display, verifying that we boarded our SkyTrain on time. Nicolette looks into my eyes, and I see her visibly shudder. She hurries away from me before I can speak.

Defeated, I head towards my seat in the back of the Train, walking past rows of noisy teens with their AirPhones out. I pass rowdy guys and loud girls, hoping that I don't get hit by someone's airborne lip gloss or something. Slipping into my assigned seats in the back-most row, 9L, I settle my weight into my aqua seat. I scan my phone against the armrest, awakening my chair's previous settings. The seat tilts to a 85 degree angle, the window fade to 100 percent transparency, and the air vents open.

I glance at Nicolette and the other girls around me, staring especially at the boys that won't quit yelling. Unfortunately, gaudy behavior slides under the radar as long as it "promotes long-lasting friendships and intimate relationships." I struggle not to roll my eyes or snicker at any conversations because I'd get a minor screening for that. The dialogue around me starts to blend together.

"Didn't youhearthat--"

"Iknowshe went to--"

"But shehadto, I mean--"

"Sucha--"

I tuck my Phone into my pocket as I try to block out everyone's voices. I can't stand all the gossiping and rowdiness people my age have, so I look out the window for a while before pulling out my Phone. At least I can pretend I'm not acompleteoutcast if I have an AirPhone in my hand.

"Hey, Ethan," Nicolette calls. I look up from my Phone as my ex-friend stands to greet her boyfriend. She embraces him and they kiss, making a lip-smacking noise as onlookers hoot.

I look away, still disapproving the approval of minimal kisses on SkyTrains. The seat next to me is empty for a reason, and I couldn't be happier. Instead of having a boy there like every other girl on this Train does, I toss my sling on the seat with a small thud. Perfect. As if on cue, a monotonous voice blares over the speakers, silencing everyone.

"Welcome to SkyTrain 267-B. Please take a seat and wait for your safety belt to fasten."

The belts on the side of my seat click around me, locking me into my seat. I shift the belt and gaze out the window, watching distant Trains on their rooftops.

"This SkyTrain will be departing in 3... 2... 1," the methodical voice counts down. The Train lurches forward, pushing me against my seat belt before releasing me to rest against the seat. Buildings in the skyline outside my window transform into a rush of blurred colors. I turn away from the window before the sight gives me a headache.

The speakers begin to recite the morning announcements. "Today is Monday, April 14, 2147. It is 53 degrees outside with pollution levels at a low 22 percent."

The SkyTrain pulls to a stop at an intersection, and the loud voice continues to list the news. A pale SkyTrain zooms by at its fast speed, jetting down the Rails towards the high school. White Trains always carry the upper-class students, or the people that'll be Screeners or leaders instead of farmers or factory workers like the rest of us.

The speakers announces a report about new construction sites as SkyTrain 267-B makes a 90-degree turn. Once the Train is fully turned, it picks up speed again. The announcements make its last statement before clicking off.

"There are approximately 160 more days until the 220th Autumnal Equinox."

A small wave of fear bounces around my stomach. I've been dreading this Equinox for a long time because it'smyEquinox year. From what little I've heard about it, my life will reduce to misery unless someone--

An ear-piercing alarm blares across the Train, the power blacking out and unlocking my safety belt. Instead of re-buckling the safety belt like everyone else does, I jerk my head to look out the window. The Train lurches, and I'm knocked out of my chair, hitting my head hard on the seat in front of me. Phones clatter to the ground, and bags launch forward.

I turn to grab my armrest as the Train shakes again, this time filling my ears with the sound of squealing breaks. My stomach jams in my throat, and students scream as my body slams against the window, now the floor, sending shots of pain through my arm. I grunt in agony as my stomach drops, somehow realizing that we're in midair. We're heading for the groundsixty stories below us.

Screams echo around the Train as I watch the pavement come closer and closer, gravity pressing on me from everywhere. My voice catches in my throat. I'm too afraid of dying to make a move towards my safety belt or my seat. Terrified out of my mind, I squeeze my eyes shut and hold onto the armrest as tightly as I can. My fingers slide and loosen their grip.

"Holland!" someone yells, sending the ear-splitting sound of crunching metal into my ears. The impact forces my fingers to let go of the armrest. I fly and hit the ceiling, banging my head so hard that I black out immediately. I presumably hit the floor, unresponsive.

-- -- -- -- --

'''

# make example sentence
sentence = Sentence(text)

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity.tag + ' -> ' + entity.text)

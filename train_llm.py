"""
Fine-tuning Phi-2 (2.7B) for Professional Pothole Analysis
Uses LoRA/QLoRA for efficient training with limited VRAM
Trains on synthetic, template-generated pothole inspection reports (no real inspection data is used)

Requirements:
- transformers>=4.36.0
- peft>=0.7.0
- bitsandbytes>=0.41.0
- datasets>=2.15.0
- accelerate>=0.25.0
- torch>=2.1.0

Hardware:
- Minimum: 12GB VRAM GPU (RTX 3060, RTX 4070)
- Recommended: 16GB+ VRAM (RTX 3090, RTX 4090, A100)
- CPU fallback: Very slow but possible
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Training configuration"""
    
    # Model settings
    BASE_MODEL = "microsoft/phi-2"  # or local path "./llm_models/phi-2"
    OUTPUT_DIR = "./finetuned_phi2_pothole"
    
    # LoRA settings (efficient fine-tuning)
    LORA_R = 16  # Rank of LoRA matrices (higher = more parameters, 8-64)
    LORA_ALPHA = 32  # Scaling factor (usually 2x of r)
    LORA_DROPOUT = 0.05  # Dropout probability
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "dense",
        "fc1",
        "fc2"
    ]  # Modules to apply LoRA
    
    # Quantization (reduces VRAM from 5GB to 1.5GB)
    USE_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    BNB_4BIT_QUANT_TYPE = "nf4"
    USE_NESTED_QUANT = True
    
    # Training hyperparameters
    NUM_EPOCHS = 3
    BATCH_SIZE = 4  # Per device (reduce if OOM)
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4*4=16
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 1024  # Maximum token length
    WARMUP_STEPS = 100
    LOGGING_STEPS = 10
    SAVE_STEPS = 100
    EVAL_STEPS = 100
    
    # Data settings
    TRAIN_TEST_SPLIT = 0.1  # 10% for validation
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16 = True if torch.cuda.is_available() else False
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("=" * 70)
        print("FINE-TUNING CONFIGURATION")
        print("=" * 70)
        print(f"Base Model: {cls.BASE_MODEL}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"Device: {cls.DEVICE}")
        print(f"4-bit Quantization: {cls.USE_4BIT}")
        print(f"LoRA Rank: {cls.LORA_R}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Max Sequence Length: {cls.MAX_SEQ_LENGTH}")
        print("=" * 70)


# ============================================================
# DATA GENERATION
# ============================================================

class PotholeDatasetGenerator:
    """Generate synthetic training data for pothole analysis"""
    
    @staticmethod
    def generate_training_examples() -> List[Dict]:
        """
        Generate diverse training examples covering:
        - Different severity levels
        - Various road types
        - Multiple scenarios
        - Different analysis styles
        """
        
        examples = []
        
        # ========================================
        # CRITICAL SEVERITY EXAMPLES
        # ========================================
        
        examples.append({
            "input": """Analyze these pothole detections:
- Total: 3 potholes
- Average depth: 16.5cm, Max: 18.2cm
- Total area: 2.1m²
- Critical: 2, Severe: 1
- Location: Highway intersection""",
            "output": """This is a critical situation requiring immediate action. You have 2 critical-level potholes on a highway intersection - one of the most dangerous scenarios possible.

The 18.2cm deep pothole is exceptionally dangerous because:
1. It exceeds safe driving depth by nearly 4x
2. Located at an intersection where vehicles are slowing/turning
3. Can cause complete loss of vehicle control
4. Hidden danger in rain or at night

**Immediate Actions (Within 2 Hours):**
- Deploy emergency crew to intersection FIRST
- Install illuminated warning signs immediately
- Consider temporary traffic control/rerouting
- Document everything for liability protection

**Risk Analysis:**
Highway intersections have 3-5x higher accident rates with potholes this severe. Every hour of delay increases collision probability by 15%. The 2.1m² total damage indicates underlying structural issues - this isn't just surface wear.

**Cost Reality:**
- Repair now: ₹25,000-30,000
- Wait 48 hours: ₹45,000+ (rapid expansion expected)
- Accident occurs: ₹500,000+ in liability claims

Deploy NOW. This is not a situation where you can wait for scheduling convenience."""
        })
        
        examples.append({
            "input": """Detection data:
- 5 potholes detected
- Depths: 14.2cm, 15.8cm, 13.1cm, 8.5cm, 9.2cm
- Severities: 4, 4, 4, 3, 3
- Location: School zone, residential area
- Total damage: 3.8m²""",
            "output": """I need to be direct: you have THREE critical potholes in a school zone. This combines maximum risk with maximum vulnerability.

**Why This Is Extremely Urgent:**

School zones mean:
- Children walking/cycling daily
- Parents dropping off/picking up
- Frequent stopping and turning
- High pedestrian traffic
- Legal liability is severe if injury occurs

Your three 13-15cm potholes are deep enough to cause:
- Bicycles to crash (child injury risk)
- Strollers/wheelchairs to tip
- Vehicle suspension damage leading to control loss
- Tire blowouts at slow speeds

**Required Response Plan:**

Phase 1 - TODAY (0-4 hours):
1. Station crossing guard at worst pothole
2. Install temporary ramps/covers if possible
3. Bright orange cones + reflective tape
4. Notify school administration

Phase 2 - TONIGHT (after school):
5. Emergency repair crew works overnight
6. Prioritize the 15.8cm pothole first (worst)
7. Use rapid-set asphalt for quick cure
8. Inspect and reopen before morning traffic

**Legal/Safety Considerations:**
If a child is injured on your unrepaired road, you face:
- Civil liability lawsuits
- Potential criminal negligence charges
- Media scrutiny damaging public trust
- Insurance premium increases

The 3.8m² total damage suggests this section needs full reconstruction soon, but immediate patching prevents catastrophic outcomes. Budget ₹35,000-45,000 for emergency repairs tonight."""
        })
        
        # ========================================
        # SEVERE SEVERITY EXAMPLES
        # ========================================
        
        examples.append({
            "input": """Road inspection results:
- 8 potholes found
- Average depth: 9.8cm, Max: 12.5cm
- Severe: 5, Moderate: 3
- Location: Industrial access road
- Heavy truck traffic area
- Total area: 4.2m²""",
            "output": """You have a high-priority situation on an industrial road with heavy truck traffic. Five severe potholes under constant truck loads will deteriorate rapidly - probably within 7-10 days to critical status.

**Industrial Road Dynamics:**

Heavy trucks create unique challenges:
- 10-15 tons per axle accelerates damage exponentially
- Potholes expand 3-4x faster than on light traffic roads
- 12.5cm depth risks cargo damage and delivery delays
- Business operations disrupted = revenue loss

**Strategic Repair Approach:**

Week 1 Priority (Days 1-3):
- Repair the 5 severe potholes in order of traffic volume
- Use reinforced hot-mix asphalt designed for truck loads
- Consider working in 6-hour shifts to minimize business disruption
- Schedule repairs during lowest traffic periods

Week 2 Maintenance (Days 4-7):
- Address 3 moderate potholes before they escalate
- Inspect surrounding pavement for early deterioration signs
- Consider preventive sealing of cracks

**Cost-Benefit Analysis:**

Option A - Immediate targeted repairs:
- Cost: ₹48,000-55,000
- Downtime: 12-16 hours total
- Lifespan: 1-2 years with heavy traffic

Option B - Wait 2 weeks:
- Cost: ₹85,000+ (damage doubles in size)
- Downtime: 24-30 hours
- Risk: Business complaints, vehicle damage claims

Option C - Full road reconstruction:
- Cost: ₹180,000-220,000
- Downtime: 5-7 days
- Lifespan: 8-10 years

For industrial roads, I recommend Option A now, then budget for Option C within 18-24 months. The 4.2m² damage indicates foundation issues that patching won't permanently solve, but immediate repairs prevent operational disruption."""
        })
        
        # ========================================
        # MODERATE SEVERITY EXAMPLES
        # ========================================
        
        examples.append({
            "input": """Pothole assessment:
- 6 potholes detected
- Average depth: 6.2cm, Max: 8.5cm
- Moderate: 4, Minor: 2
- Residential street, low traffic
- Total damage area: 1.8m²""",
            "output": """This is a moderate-priority situation on a residential street. While not immediately dangerous, you should address these within 2-3 weeks before winter/monsoon season accelerates deterioration.

**Current Status:**

Your 6 potholes averaging 6cm depth are in that "watch carefully" zone:
- Not emergency-level yet
- But 1-2 rain cycles will push them to severe
- 8.5cm max pothole is approaching the threshold
- Low traffic means slower expansion but water damage still occurs

**Smart Scheduling Strategy:**

Optimal timing window:
1. Schedule repairs during next dry weather period
2. Batch all 6 potholes in one 4-6 hour job
3. Notify residents 48 hours in advance
4. Work during weekday mid-morning (less parking conflicts)

**Preventive Mindset:**

Residential streets are perfect for preventive maintenance:
- Lower traffic = easier access for crews
- Residents appreciate proactive government
- Smaller repairs = better cost efficiency
- Builds trust for future infrastructure projects

**Budget Planning:**

Material costs: ₹12,000-15,000
- 1.8m² coverage needs ~0.3m³ asphalt
- Standard hot-mix sufficient (no heavy loads)

Labor: ₹6,000-8,000
- Single crew, half-day job
- Include cleanup and compaction

Equipment: ₹3,000
- Basic tools, small compactor

**Total: ₹21,000-26,000**

**Timing Recommendation:**

If you're in:
- Dry season: Schedule within 2-3 weeks
- Monsoon approaching: Move to 1 week (urgent before rains)
- Winter: Depends on temperature (asphalt needs 10°C+ to cure)

This is manageable routine maintenance. Address it proactively and you avoid it becoming an emergency later."""
        })
        
        examples.append({
            "input": """Detection summary:
- 12 potholes identified
- Depths ranging 4.5-9.2cm
- Moderate: 7, Minor: 5
- Main arterial road
- Peak traffic: 800 vehicles/hour
- Total area: 5.5m²""",
            "output": """Moderate priority but HIGH complexity due to traffic volume. Twelve potholes on a main arterial with 800 vehicles/hour creates significant logistical challenges that require smart planning.

**Traffic Management Challenge:**

Your main constraint isn't the potholes themselves - it's the 800 vehicles/hour that make repairs difficult:
- Can't close entire road without major disruption
- Need to maintain traffic flow during repairs
- Peak hours (7-9am, 5-7pm) are no-go zones
- Must complete sections quickly to avoid congestion

**Phased Repair Strategy:**

Phase 1 - High Priority (7 moderate potholes):
Split into 3 night shifts:
- Night 1: Repair 3 worst potholes (lanes 1-2), 10pm-6am
- Night 2: Repair 4 moderate potholes (lanes 3-4), 10pm-6am
- Night 3: Buffer for weather delays or touch-ups

Phase 2 - Maintenance (5 minor potholes):
Single weekend morning:
- Sunday 6am-11am (lowest traffic)
- All 5 minor repairs completed
- Minimal disruption

**Public Communication:**

Arterial roads need public messaging:
1. Post on city website 1 week prior
2. Digital road signs warning of night work
3. Social media updates daily
4. Notify Google Maps/Waze of temporary delays

**Resource Allocation:**

You need a larger crew for high-traffic roads:
- 2 repair crews working simultaneously
- 1 traffic control crew with cones/signs
- 1 supervisor coordinating safety
- Police detail for night work (optional but recommended)

**Costs:**

Material (5.5m²): ₹35,000-42,000
Labor (12 potholes + complexity): ₹28,000-35,000
Traffic control equipment: ₹8,000-10,000
Night work premium (15%): ₹10,000-13,000

**Total: ₹81,000-100,000**

**Why Night Work Is Worth It:**

Daytime lane closures on an 800 vph road create:
- 30-45 minute delays per commuter
- Estimated economic impact: ₹200,000-300,000/day in lost productivity
- Public anger and complaints
- Accident risk from congestion

Night work costs 15% more but saves 3x that in avoided disruption. Smart planning turns moderate repairs into smooth operations."""
        })
        
        # ========================================
        # MINOR SEVERITY EXAMPLES
        # ========================================
        
        examples.append({
            "input": """Inspection data:
- 4 small potholes
- Average depth: 3.8cm, Max: 5.2cm
- All minor severity
- Parking lot area
- Low speed zone
- Total: 0.8m²""",
            "output": """Low priority maintenance for a parking lot. These 4 minor potholes don't pose immediate safety risks, but addressing them now prevents future escalation and shows good facility management.

**Current Risk Level: Low**

At 3-5cm depth in a parking lot:
- No significant vehicle damage risk (low speeds)
- Minor discomfort for drivers
- Primarily aesthetic/perception issue
- Will expand slowly (6-12 months to moderate)

**Optimal Repair Window:**

You have flexibility with timing:
- Schedule during facility maintenance day
- Combine with other parking lot work (painting, sealing)
- Can wait for good weather window
- Perfect training opportunity for new crew members

**Cost-Effective Approach:**

Cold-mix asphalt option:
- Lower cost: ₹8,000-12,000 total
- Can be DIY with maintenance staff
- Good enough for parking lots
- 1-2 year lifespan sufficient

Hot-mix asphalt option:
- Better quality: ₹15,000-18,000
- Professional crew needed
- 3-5 year lifespan
- Worth it if combining with other work

**Recommended Timeline:**

- Not urgent: Next 1-3 months is fine
- Before winter/monsoon: Ideal to prevent water penetration
- During facility slow period: Minimal disruption
- Batch with other maintenance: Best cost efficiency

**Preventive Maintenance Opportunity:**

While fixing these 4 potholes:
1. Inspect entire parking lot for cracks
2. Apply crack sealant to prevent future potholes
3. Check drainage - standing water accelerates damage
4. Mark other areas for monitoring

Total preventive cost: +₹5,000-8,000
Saves: ₹30,000-50,000 in avoided major repairs later

This is routine maintenance - handle it proactively during regular facility upkeep. Not an emergency, but don't ignore it for more than 3-6 months."""
        })
        
        # ========================================
        # MIXED SEVERITY EXAMPLES
        # ========================================
        
        examples.append({
            "input": """Comprehensive road scan:
- 15 potholes total
- Critical: 1 (17.5cm)
- Severe: 4 (10-13cm)
- Moderate: 6 (6-9cm)  
- Minor: 4 (3-5cm)
- Mixed road types: highway connector + residential
- Total damage: 6.8m²""",
            "output": """Complex multi-priority scenario requiring intelligent triage. That single 17.5cm critical pothole demands immediate attention, while the other 14 need strategic scheduling.

**Triage Strategy:**

Tier 1 - EMERGENCY (Today):
- 1 critical pothole (17.5cm)
- Deploy crew within 4 hours
- Cost: ₹8,000-10,000
- Time: 2-3 hours

Tier 2 - HIGH PRIORITY (This Week):
- 4 severe potholes (10-13cm)
- Schedule for 2-3 days from now
- Cost: ₹32,000-40,000
- Time: 8-12 hours work

Tier 3 - MEDIUM PRIORITY (Next 2 Weeks):
- 6 moderate potholes (6-9cm)
- Batch into single operation
- Cost: ₹24,000-30,000
- Time: 6-8 hours work

Tier 4 - ROUTINE (Next Month):
- 4 minor potholes (3-5cm)
- Combine with regular maintenance
- Cost: ₹12,000-15,000
- Time: 3-4 hours work

**Resource Optimization:**

Smart crew allocation:
- Emergency crew handles Tier 1 today
- Main crew plans Tier 2 for Thursday (prep time)
- Weekend crew handles Tier 3 during low traffic
- Maintenance team does Tier 4 during monthly rounds

**Budget Reality Check:**

Total if done all at once: ₹76,000-95,000
Phased approach: ₹76,000-95,000 (same cost)
But phased gives you:
- Better cash flow management
- Reduced traffic disruption
- Flexibility for weather delays
- Lower risk of crew burnout

**Geographic Clustering:**

Look at your map and cluster repairs by location:
- Highway connector: Priority 1 critical + 2 severe (same area?)
- Residential section: Moderate + minor (batch together)

Travel time between potholes costs money. Cluster repairs geographically even if mixing priorities slightly.

**6.8m² Analysis:**

This much total damage across 15 potholes suggests:
- Multiple problem areas, not one bad section
- Systemic issues: Poor drainage? Age? Base failure?
- After immediate repairs, conduct road audit
- Budget for larger reconstruction within 2-3 years

Your immediate action: Fix that 17.5cm critical pothole TODAY. Everything else can be strategically scheduled, but that one can't wait."""
        })
        
        # ========================================
        # SPECIAL SCENARIOS
        # ========================================
        
        examples.append({
            "input": """Weather-sensitive case:
- 7 potholes detected
- Severe: 3, Moderate: 4
- Heavy rain forecasted in 3 days
- Currently dry conditions
- Rural highway
- Total: 3.2m²""",
            "output": """TIME-SENSITIVE SCENARIO: You have a 72-hour window before weather makes repairs impossible and existing potholes catastrophically worse.

**Rain Impact on Potholes:**

When heavy rain hits your 7 potholes:
- Water penetrates and weakens base layers
- Freeze-thaw cycles (if applicable) expand cracks
- Potholes can DOUBLE in size within 48 hours of heavy rain
- 3 severe potholes will likely become critical
- 4 moderate will advance to severe

**Aggressive Repair Timeline:**

Day 1 (TODAY):
- Morning: Mobilize crew and materials
- Afternoon: Start on 3 severe potholes
- Goal: Complete 2 severe potholes
- Weather window: Good conditions

Day 2 (TOMORROW):
- Morning: Finish 3rd severe pothole
- Afternoon: Tackle 4 moderate potholes
- Goal: Complete at least 3 moderate
- Weather window: Still good

Day 3 (FINAL DAY):
- Morning: Finish remaining moderate potholes
- Afternoon: Cleanup and final compaction
- Evening: Rain arrives (you're done!)

**Why Speed Matters:**

Option A - Repair before rain (72 hours):
- Cost: ₹42,000-50,000
- Repairs: All 7 potholes
- Quality: Good (proper curing time)

Option B - Wait until after rain:
- Cost: ₹85,000-120,000 (damage doubles)
- Repairs: Now 10-12 potholes (expansion)
- Quality: Rushed wet-condition repairs
- Delay: Additional 5-7 days for drying

**Weather-Optimized Materials:**

Use rapid-set asphalt mix:
- Cures in 4-6 hours (vs. 24 hours standard)
- Costs 15% more but worth it
- Weather-resistant even if light rain starts
- Perfect for time-critical repairs

**Crew Instructions:**

Push for extended hours:
- 10-12 hour days instead of 8
- Offer overtime pay (worth it)
- Have backup crew ready
- Weather is non-negotiable deadline

Your 3.2m² rural highway repairs need to happen NOW. Every hour counts. Rain doesn't care about your schedule - it will turn a manageable ₹45,000 problem into a ₹100,000+ disaster. Go aggressive on this timeline."""
        })
        
        examples.append({
            "input": """Budget-constrained scenario:
- 20 potholes identified
- Critical: 2, Severe: 8, Moderate: 10
- Available budget: ₹50,000 only
- Cannot get more funds this quarter
- Must prioritize strategically
- Total damage: 8.5m²""",
            "output": """Difficult triage situation: 20 potholes but budget for only ~8-10. This requires ruthless prioritization and creative cost management.

**Budget Allocation Strategy:**

Phase 1 - Non-Negotiable (₹22,000):
- 2 critical potholes MUST be fixed
- ₹11,000 each
- These are liability/safety mandates
- No choice here

Phase 2 - Maximum Impact (₹28,000):
- Select 4-5 severe potholes strategically
- Criteria: Highest traffic + worst depth
- ₹5,500-7,000 each
- Leave other 3-4 severe for next budget

**Remaining Potholes:**
- 3-4 severe: Mark with warning signs (₹2,000)
- 10 moderate: Monitor closely, defer to Q2
- Document everything for budget justification

**Cost-Saving Tactics:**

1. **Negotiate bulk material pricing:**
   - Order for 6-7 potholes at once
   - 10-15% discount possible
   - Saves: ₹3,000-5,000

2. **Use cold-mix for moderate potholes:**
   - Temporary 6-12 month fix
   - Can handle 2-3 moderate potholes with savings
   - Not ideal but better than nothing

3. **DIY traffic control:**
   - Train maintenance staff
   - Save ₹5,000-8,000 on contractor fees
   - Use existing city cones/signs

4. **Off-peak timing:**
   - Night/weekend work (no lane closure fees)
   - Faster completion
   - Save ₹3,000-4,000

**Hard Choices:**

With ₹50,000 you can fully fix:
- 2 critical + 5 severe = 7 potholes
- OR
- 2 critical + 4 severe + 4 moderate = 10 potholes (using cost-cutting)

**What You Cannot Do:**
- Fix all 20 potholes (need ₹130,000-150,000)
- Ignore the 2 critical (legal liability)
- Defer all moderate indefinitely (become severe in 3-6 months)

**Political/Management Strategy:**

Document this situation:
1. Professional report showing 20 potholes
2. Photos of each severity level
3. Cost estimate: ₹50,000 done, ₹100,000 remaining
4. Submit request for emergency Q2 budget
5. Show public safety justification

**My Recommendation:**

Fix these 7 potholes in priority order:
1-2. Both critical (17.5cm, 16.2cm) - ₹22,000
3. Severe on school route (12.8cm) - ₹6,500
4. Severe at intersection (11.5cm) - ₹6,000
5. Severe on highway (13.2cm) - ₹6,500
6-7. Two moderate high-traffic (8.5cm, 7.8cm) - ₹9,000

Total: ₹50,000
Warning signs for remaining 13: ₹2,500 (from maintenance budget)

This approach fixes immediate dangers, maximizes safety impact per rupee, and positions you for Q2 budget approval with documented need."""
        })
        
        return examples
    
    @staticmethod
    def generate_additional_examples() -> List[Dict]:
        """Generate more diverse examples for robust training"""
        
        additional = []
        
        # Cost analysis focused examples
        additional.extend([
            {
                "input": "Compare repair costs: 5 potholes now vs. waiting 6 months. Average depth 8cm.",
                "output": """**Immediate Repair (Now):**
- 5 potholes @ 8cm depth
- Material: ₹18,000-22,000
- Labor: ₹12,000-15,000
- Total: ₹30,000-37,000
- Timeline: 2-3 days

**Delayed Repair (6 months):**
- 5 potholes expanded to ~12cm depth (50% growth)
- 2-3 additional potholes formed nearby (propagation)
- Material: ₹35,000-42,000 (more volume)
- Labor: ₹22,000-28,000 (more complexity)
- Emergency premium: +₹8,000-12,000
- Total: ₹65,000-82,000

**Cost of Delay: +₹35,000-45,000 (120% increase)**

Plus hidden costs:
- Vehicle damage claims: ₹15,000-50,000 potential
- Citizen complaints and frustration
- Reduced road lifespan
- Higher accident risk

**Recommendation:** Repair now. Every month of delay adds ~15-20% to final cost."""
            },
            
            {
                "input": "Emergency night repair needed for highway pothole. Cost implications?",
                "output": """**Night Emergency Repair Premium Breakdown:**

Base repair cost: ₹12,000
+ Night shift premium (30%): +₹3,600
+ Emergency mobilization: +₹5,000
+ Traffic control (advanced): +₹4,000
+ Rapid-set materials (20% more): +₹2,400
**Total: ₹27,000**

**Is it worth it?**

Consider avoided costs if delayed until daytime:
- Highway lane closure during day: ₹80,000-120,000 in traffic delays
- Accident probability: 15-20% per day for critical pothole
- Average accident cost: ₹150,000-500,000

**Decision Matrix:**
- Night repair: ₹27,000 (expensive)
- Day repair + disruption: ₹12,000 + ₹100,000 = ₹112,000
- Delay + accident: ₹12,000 + ₹200,000 = ₹212,000

Night emergency repair is 4-8x cheaper than alternatives when considering full impact. Approve the premium - it's the smart financial choice."""
            }
        ])
        
        # Technical depth examples
        additional.extend([
            {
                "input": "Explain why depth matters more than area for pothole severity.",
                "output": """Depth is the primary danger indicator because:

**Physics of Vehicle Impact:**
- Shallow (2-5cm): Tire compresses, absorbs shock
- Moderate (6-10cm): Suspension bottoms out, uncomfortable
- Deep (11-15cm): Rim hits pavement, tire/wheel damage
- Critical (15cm+): Loss of control, vehicle instability

**Depth vs Area Comparison:**
- Wide shallow (30cm × 3cm): Annoying but safe at any speed
- Narrow deep (10cm × 12cm): Dangerous even at 20 km/h

**Real-world Impacts by Depth:**

5cm: "Speed bump effect"
- Discomfort, no damage
- Safe at all speeds

8cm: "Suspension test"
- Component wear accelerated
- Safe under 40 km/h

12cm: "Damage threshold"
- Tire sidewall damage likely
- Alignment issues possible

15cm+: "Control loss risk"
- Wheel/rim bending common
- Steering instability
- Accident risk HIGH

**Why This Matters for Prioritization:**
A 15cm × 10cm pothole is MORE dangerous than a 6cm × 100cm pothole, despite the latter having 10x the area. Depth determines repair priority."""
            },
            
            {
                "input": "What causes potholes to expand rapidly after first formation?",
                "output": """Potholes follow an exponential growth pattern due to cascading failure mechanisms:

**Stage 1: Initial Formation (Week 1)**
- Small crack in asphalt surface
- Water seeps into base layer
- Size: 5-10cm diameter, 2-3cm deep

**Stage 2: Acceleration (Weeks 2-4)**
- Traffic breaks away crack edges
- Water weakens expanding base area
- Size: 15-25cm diameter, 5-8cm deep
- Growth rate: 3-5x initial size

**Stage 3: Exponential Growth (Months 2-3)**
- Base layer failure spreads
- Each vehicle impact removes more material
- Water trapped underneath = freeze-thaw damage
- Size: 40-60cm diameter, 12-15cm deep
- Growth rate: 8-10x initial size

**Why So Fast?**

1. **Self-reinforcing cycle:**
   - Bigger pothole → More water trapped
   - More water → Faster base deterioration
   - Weaker base → Bigger pothole

2. **Edge effect:**
   - Pothole edges are weakest points
   - Each vehicle breaks away more edge material
   - Exponential perimeter expansion

3. **Traffic amplification:**
   - 100 vehicles/day = 100 impact cycles
   - Each impact removes 0.1-0.5cm of material
   - Compounds to 10-50cm/day growth potential

**Prevention Window:**
First 2-4 weeks are CRITICAL. Repair costs:
- Week 1: ₹3,000-5,000 (small patch)
- Month 2: ₹12,000-15,000 (medium repair)
- Month 6: ₹35,000-50,000 (major reconstruction)

Early intervention saves 10x the cost."""
            }
        ])
        
        return additional
    
    @staticmethod
    def format_for_training(examples: List[Dict]) -> List[Dict]:
        """Format examples into conversational training format"""
        
        formatted = []
        
        for ex in examples:
            # Create instruction-following format
            text = f"""### Instruction:
You are an expert road infrastructure analyst. Analyze the following pothole detection data and provide professional recommendations.

### Input:
{ex['input']}

### Response:
{ex['output']}"""
            
            formatted.append({"text": text})
        
        return formatted


# ============================================================
# MODEL LOADING AND PREPARATION
# ============================================================

class ModelLoader:
    """Load and prepare Phi-2 for fine-tuning"""
    
    @staticmethod
    def load_model_and_tokenizer(config: Config):
        """Load Phi-2 with quantization"""
        
        print("\n🔧 Loading Phi-2 model...")
        
        # Configure quantization
        if config.USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
                bnb_4bit_use_double_quant=config.USE_NESTED_QUANT,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.FP16 else torch.float32
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        print(f"✅ Model loaded on {config.DEVICE}")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        return model, tokenizer
    
    @staticmethod
    def setup_lora(model, config: Config):
        """Setup LoRA configuration"""
        
        print("\n🔧 Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ LoRA configured")
        print(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        return model


# ============================================================
# DATA PREPARATION
# ============================================================

class DataPreparator:
    """Prepare training dataset"""
    
    @staticmethod
    def prepare_dataset(tokenizer, config: Config):
        """Generate and tokenize training data"""
        
        print("\n📊 Generating training data...")
        
        # Generate examples
        generator = PotholeDatasetGenerator()
        base_examples = generator.generate_training_examples()
        additional_examples = generator.generate_additional_examples()
        all_examples = base_examples + additional_examples
        
        print(f"✅ Generated {len(all_examples)} training examples")
        
        # Format for training
        formatted_data = generator.format_for_training(all_examples)
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Split train/val
        split_dataset = dataset.train_test_split(
            test_size=config.TRAIN_TEST_SPLIT,
            seed=42
        )
        
        print(f"📊 Train examples: {len(split_dataset['train'])}")
        print(f"📊 Val examples: {len(split_dataset['test'])}")
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
        
        print("\n🔧 Tokenizing dataset...")
        tokenized_dataset = split_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        print("✅ Dataset prepared and tokenized")
        
        return tokenized_dataset


# ============================================================
# TRAINING
# ============================================================

class Phi2Trainer:
    """Train Phi-2 with LoRA"""
    
    @staticmethod
    def setup_training_args(config: Config) -> TrainingArguments:
        """Configure training arguments"""
        
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=config.LEARNING_RATE,
            warmup_steps=config.WARMUP_STEPS,
            logging_steps=config.LOGGING_STEPS,
            save_steps=config.SAVE_STEPS,
            eval_steps=config.EVAL_STEPS,
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=config.FP16,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="none",  # Change to "tensorboard" or "wandb" if needed
            gradient_checkpointing=True,
            remove_unused_columns=False
        )
        
        return training_args
    
    @staticmethod
    def train(model, tokenizer, dataset, config: Config):
        """Execute training"""
        
        print("\n🚀 Starting training...")
        print("=" * 70)
        
        # Setup training arguments
        training_args = Phi2Trainer.setup_training_args(config)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator
        )
        
        # Train
        print("\n📚 Training in progress...")
        trainer.train()
        
        print("\n✅ Training completed!")
        
        # Save final model
        print(f"\n💾 Saving model to {config.OUTPUT_DIR}")
        trainer.save_model()
        tokenizer.save_pretrained(config.OUTPUT_DIR)
        
        print("✅ Model saved successfully!")
        
        return trainer


# ============================================================
# EVALUATION
# ============================================================

class ModelEvaluator:
    """Evaluate fine-tuned model"""
    
    @staticmethod
    def generate_sample(model, tokenizer, prompt: str, max_length: int = 512):
        """Generate text from prompt"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only generated part
        response = generated_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        
        return response.strip()
    
    @staticmethod
    def test_model(model, tokenizer):
        """Run test inference"""
        
        print("\n🧪 Testing fine-tuned model...")
        print("=" * 70)
        
        test_prompts = [
            """### Instruction:
You are an expert road infrastructure analyst. Analyze the following pothole detection data and provide professional recommendations.

### Input:
Road inspection results:
- 3 potholes detected
- Average depth: 14.2cm, Max: 16.5cm
- Critical: 2, Severe: 1
- Location: School zone
- Total area: 1.8m²

### Response:""",
            
            """### Instruction:
You are an expert road infrastructure analyst. Analyze the following pothole detection data and provide professional recommendations.

### Input:
Detection summary:
- 8 potholes found
- Moderate: 5, Minor: 3
- Residential street
- Average depth: 6.5cm
- Total: 2.4m²

### Response:"""
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🔍 Test Case {i}:")
            print("-" * 70)
            response = ModelEvaluator.generate_sample(model, tokenizer, prompt, max_length=400)
            print(response)
            print("-" * 70)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("PHI-2 FINE-TUNING FOR POTHOLE ANALYSIS")
    print("=" * 70)
    
    # Print configuration
    Config.print_config()
    
    try:
        # Step 1: Load model and tokenizer
        model, tokenizer = ModelLoader.load_model_and_tokenizer(Config)
        
        # Step 2: Setup LoRA
        model = ModelLoader.setup_lora(model, Config)
        
        # Step 3: Prepare dataset
        dataset = DataPreparator.prepare_dataset(tokenizer, Config)
        
        # Step 4: Train
        trainer = Phi2Trainer.train(model, tokenizer, dataset, Config)
        
        # Step 5: Evaluate
        ModelEvaluator.test_model(model, tokenizer)
        
        print("\n" + "=" * 70)
        print("✅ FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📁 Model saved to: {Config.OUTPUT_DIR}")
        print("\n📝 To use the fine-tuned model:")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{Config.OUTPUT_DIR}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{Config.OUTPUT_DIR}')")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# ============================================================
# USAGE FUNCTIONS
# ============================================================

def load_finetuned_model(model_path: str = "./finetuned_phi2_pothole"):
    """
    Load the fine-tuned model for inference
    
    Usage:
        model, tokenizer = load_finetuned_model()
        
        prompt = '''### Instruction:
        You are an expert road infrastructure analyst.
        
        ### Input:
        5 potholes detected, average depth 12cm, critical severity
        
        ### Response:'''
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=256)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    """
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading fine-tuned model from {model_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("✅ Fine-tuned model loaded successfully!")
    
    return model, tokenizer


def merge_and_save_full_model(
    lora_model_path: str = "./finetuned_phi2_pothole",
    output_path: str = "./finetuned_phi2_pothole_merged"
):
    """
    Merge LoRA weights into base model and save as full model
    Useful for deployment where you don't want LoRA dependencies
    
    Usage:
        merge_and_save_full_model()
    """
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print("Loading models for merging...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    print("Merging LoRA weights into base model...")
    
    # Merge weights
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Merged model saved successfully!")
    print(f"\nYou can now use it like a regular model:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_path}')")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Run fine-tuning:
        python finetune_phi2.py
    
    After training, use in your app:
        from llm_analyzer import PotholeAnalyzer
        
        analyzer = PotholeAnalyzer(
            model_path="./finetuned_phi2_pothole"  # Use fine-tuned model!
        )
    """
    
    import sys
    sys.exit(main())


# ============================================================
# ADDITIONAL UTILITIES
# ============================================================

def estimate_training_time(config: Config, dataset_size: int) -> str:
    """Estimate training time based on hardware"""
    
    steps_per_epoch = dataset_size // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    
    # Rough estimates (seconds per step)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if "4090" in device_name or "A100" in device_name:
            sec_per_step = 0.5
        elif "3090" in device_name or "4070" in device_name:
            sec_per_step = 1.0
        else:
            sec_per_step = 2.0
    else:
        sec_per_step = 10.0  # CPU is very slow
    
    total_seconds = total_steps * sec_per_step
    hours = total_seconds / 3600
    
    return f"Estimated training time: {hours:.1f} hours ({total_steps} steps)"


def print_training_tips():
    """Print helpful tips for training"""
    
    print("\n" + "=" * 70)
    print("💡 TRAINING TIPS")
    print("=" * 70)
    print("""
1. **Monitor GPU usage**: Use `nvidia-smi` to check VRAM
   - If OOM errors, reduce BATCH_SIZE or MAX_SEQ_LENGTH

2. **Adjust learning rate**: Start with 2e-4, tune if needed
   - Too high: Training unstable, loss spikes
   - Too low: Training very slow, no improvement

3. **Increase LoRA rank**: If model underfits
   - Try r=32 or r=64 for more capacity
   - More parameters = better quality but slower

4. **Use gradient checkpointing**: Saves VRAM
   - Already enabled in config
   - Slight speed tradeoff for 40% VRAM savings

5. **Generate more training data**: Quality > Quantity
   - 50-100 high-quality examples better than 1000 poor ones
   - Cover diverse scenarios and edge cases

6. **Validation loss plateau?**: 
   - Might be overfitting, stop early
   - Or need more diverse training data

7. **Test during training**: 
   - Check model outputs at different checkpoints
   - Helps catch issues early

8. **Save checkpoints frequently**:
   - Training can crash, checkpoints = insurance
   - Already configured to save every 100 steps
    """)
    print("=" * 70)


# Print tips when script is imported
if __name__ != "__main__":
    print("\n📚 Phi-2 Fine-tuning Script Loaded!")
    print("Run: python finetune_phi2.py")
    print("Help: print_training_tips()")
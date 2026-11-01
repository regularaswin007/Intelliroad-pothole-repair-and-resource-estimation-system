"""
Dual Analyzer System: LLM-Powered + Fast Rule-Based
Choose based on your needs: Quality vs Speed
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


class SmartPotholeAnalyzer:
    """
    Advanced rule-based analyzer with professional output
    NO LLM - Instant results (<0.1 seconds)
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.analysis_count = 0
        print("✅ Smart Analyzer initialized (instant mode)")
    
    def analyze_detections(self, detections, image_size=None):
        """
        Generate comprehensive professional analysis instantly
        
        Args:
            detections: List of detection dicts from cv_detection
            image_size: Optional tuple (width, height) for context
            
        Returns:
            dict with detailed analysis results
        """
        self.analysis_count += 1
        
        if not detections or len(detections) == 0:
            return self._generate_no_detection_report()
        
        # Compute detailed statistics
        stats = self._compute_detailed_stats(detections)
        
        # Generate comprehensive analysis
        analysis = {
            'summary': self._generate_summary(stats),
            'severity_report': self._generate_severity_report(stats),
            'recommendations': self._generate_recommendations(stats),
            'risk_assessment': self._generate_risk_assessment(stats),
            'cost_estimate': self._generate_cost_estimate(stats),
            'overall_severity': stats['overall_severity'],
            'urgency': stats['urgency'],
            'risk_level': stats['risk_level'],
            'stats': stats
        }
        
        return analysis
    
    def _compute_detailed_stats(self, detections):
        """Extract comprehensive statistics"""
        depths = [d['depth'] for d in detections]
        areas = [d['area'] for d in detections]
        severities = [d['severity'] for d in detections]
        methods = [d.get('method', 'Unknown') for d in detections]
        
        # Severity counts
        critical = sum(1 for s in severities if s == 4)
        severe = sum(1 for s in severities if s == 3)
        moderate = sum(1 for s in severities if s == 2)
        minor = sum(1 for s in severities if s == 1)
        
        # Depth analysis
        avg_depth = np.mean(depths)
        max_depth = np.max(depths)
        min_depth = np.min(depths)
        depth_std = np.std(depths)
        
        # Area analysis
        total_area = np.sum(areas)
        avg_area = np.mean(areas)
        max_area = np.max(areas)
        
        # Determine overall severity and urgency
        if critical > 0:
            overall_severity = "CRITICAL"
            urgency = "IMMEDIATE (within 24 hours)"
            risk_level = "EXTREME"
            priority_score = 10
        elif severe >= 3 or (severe >= 1 and max_depth > 15):
            overall_severity = "SEVERE"
            urgency = "HIGH (within 2-3 days)"
            risk_level = "HIGH"
            priority_score = 8
        elif severe > 0 or moderate >= 4:
            overall_severity = "MODERATE"
            urgency = "MEDIUM (within 1-2 weeks)"
            risk_level = "MODERATE"
            priority_score = 6
        else:
            overall_severity = "MINOR"
            urgency = "LOW (routine maintenance)"
            risk_level = "LOW"
            priority_score = 3
        
        # Detection quality
        detection_methods = {}
        for method in methods:
            detection_methods[method] = detection_methods.get(method, 0) + 1
        
        return {
            'total_count': len(detections),
            'avg_depth': avg_depth,
            'max_depth': max_depth,
            'min_depth': min_depth,
            'depth_std': depth_std,
            'avg_area': avg_area,
            'max_area': max_area,
            'total_area': total_area,
            'critical_count': critical,
            'severe_count': severe,
            'moderate_count': moderate,
            'minor_count': minor,
            'overall_severity': overall_severity,
            'urgency': urgency,
            'risk_level': risk_level,
            'priority_score': priority_score,
            'detection_methods': detection_methods,
            'detections': detections
        }
    
    def _generate_summary(self, stats):
        """Generate executive summary"""
        summary = f"**Road Inspection Report - Analysis #{self.analysis_count}**\n\n"
        
        summary += f"🔍 Detected **{stats['total_count']} pothole(s)** with **{stats['overall_severity']}** condition severity.\n\n"
        
        summary += f"📏 **Depth Analysis:**\n"
        summary += f"   • Average: {stats['avg_depth']:.1f} cm\n"
        summary += f"   • Maximum: {stats['max_depth']:.1f} cm\n"
        summary += f"   • Range: {stats['min_depth']:.1f} - {stats['max_depth']:.1f} cm\n\n"
        
        summary += f"📐 **Surface Damage:**\n"
        summary += f"   • Total area affected: {stats['total_area']:.2f} m²\n"
        summary += f"   • Average pothole size: {stats['avg_area']:.3f} m²\n"
        summary += f"   • Largest pothole: {stats['max_area']:.3f} m²\n\n"
        
        # Add warnings for critical conditions
        if stats['max_depth'] > 15:
            summary += f"⚠️ **WARNING:** Deepest pothole ({stats['max_depth']:.1f}cm) exceeds safety threshold!\n\n"
        
        if stats['total_area'] > 5.0:
            summary += f"⚠️ **NOTICE:** Large cumulative damage area ({stats['total_area']:.1f}m²) may require resurfacing.\n\n"
        
        return summary
    
    def _generate_severity_report(self, stats):
        """Generate detailed severity breakdown"""
        report = f"**Risk Classification: {stats['risk_level']}**\n\n"
        
        report += f"📊 **Severity Distribution:**\n\n"
        
        total = stats['total_count']
        
        if stats['critical_count'] > 0:
            pct = (stats['critical_count'] / total) * 100
            report += f"🔴 **Critical (Level 4):** {stats['critical_count']} pothole(s) ({pct:.0f}%)\n"
            report += f"   → Immediate safety hazard, vehicle damage risk\n\n"
        
        if stats['severe_count'] > 0:
            pct = (stats['severe_count'] / total) * 100
            report += f"🟠 **Severe (Level 3):** {stats['severe_count']} pothole(s) ({pct:.0f}%)\n"
            report += f"   → High priority, rapid deterioration expected\n\n"
        
        if stats['moderate_count'] > 0:
            pct = (stats['moderate_count'] / total) * 100
            report += f"🟡 **Moderate (Level 2):** {stats['moderate_count']} pothole(s) ({pct:.0f}%)\n"
            report += f"   → Requires attention, monitor for progression\n\n"
        
        if stats['minor_count'] > 0:
            pct = (stats['minor_count'] / total) * 100
            report += f"🟢 **Minor (Level 1):** {stats['minor_count']} pothole(s) ({pct:.0f}%)\n"
            report += f"   → Low priority, routine maintenance cycle\n\n"
        
        report += f"---\n"
        report += f"**Overall Assessment:** {stats['overall_severity']} condition\n"
        report += f"**Priority Score:** {stats['priority_score']}/10"
        
        return report
    
    def _generate_recommendations(self, stats):
        """Generate actionable recommendations"""
        rec = f"**Action Plan - {stats['urgency']}**\n\n"
        
        # Immediate actions
        if stats['critical_count'] > 0:
            rec += f"🚨 **URGENT ACTIONS REQUIRED:**\n\n"
            rec += f"1. **Emergency Response:** Deploy repair crew immediately for {stats['critical_count']} critical pothole(s)\n"
            rec += f"2. **Safety Measures:** Install warning signs/barriers around hazardous areas\n"
            rec += f"3. **Traffic Management:** Consider lane closures if needed for public safety\n"
            rec += f"4. **Documentation:** Photograph all critical potholes for liability protection\n\n"
        
        # Priority repairs
        if stats['severe_count'] > 0:
            rec += f"⚡ **HIGH PRIORITY REPAIRS:**\n\n"
            rec += f"• Schedule {stats['severe_count']} severe pothole repair(s) within 2-3 days\n"
            rec += f"• Use hot-mix asphalt for permanent repairs\n"
            rec += f"• Prioritize high-traffic areas first\n\n"
        
        # Maintenance planning
        if stats['moderate_count'] + stats['minor_count'] > 0:
            rec += f"📋 **MAINTENANCE SCHEDULE:**\n\n"
            rec += f"• Plan routine repairs for {stats['moderate_count'] + stats['minor_count']} moderate/minor pothole(s)\n"
            rec += f"• Batch repairs for cost efficiency\n"
            rec += f"• Schedule during low-traffic periods\n\n"
        
        # Resource allocation
        rec += f"💼 **RESOURCE ALLOCATION:**\n\n"
        
        # Estimate crew requirements
        if stats['total_count'] <= 5:
            rec += f"• Recommended: 1 repair crew (2-3 workers)\n"
        elif stats['total_count'] <= 15:
            rec += f"• Recommended: 2 repair crews (4-6 workers)\n"
        else:
            rec += f"• Recommended: 3+ repair crews (6+ workers) for parallel operations\n"
        
        # Material estimate
        estimated_material = stats['total_area'] * 0.15  # rough estimate with overfill
        rec += f"• Estimated material needed: {estimated_material:.2f} m³ of asphalt\n"
        
        # Special considerations
        if stats['total_area'] > 10.0:
            rec += f"\n💡 **SPECIAL RECOMMENDATION:**\n"
            rec += f"Large damaged area ({stats['total_area']:.1f}m²) detected. Consider full road resurfacing instead of patching for better long-term results and cost efficiency.\n"
        
        # DRL integration
        rec += f"\n🤖 **AI-OPTIMIZED SCHEDULING:**\n"
        rec += f"Use the DRL priority system in 'Simulate Repairs' mode to optimize:\n"
        rec += f"• Repair sequence for maximum safety impact\n"
        rec += f"• Travel routes to minimize equipment movement\n"
        rec += f"• Material allocation for efficient resource usage\n"
        
        return rec
    
    def _generate_risk_assessment(self, stats):
        """Generate risk assessment"""
        risk = f"**Risk Level: {stats['risk_level']}**\n\n"
        
        # Safety risks
        risk += f"⚠️ **Safety Risks:**\n"
        if stats['critical_count'] > 0:
            risk += f"• HIGH: {stats['critical_count']} critical pothole(s) pose immediate accident risk\n"
        if stats['max_depth'] > 12:
            risk += f"• HIGH: Deep potholes ({stats['max_depth']:.1f}cm) can cause tire damage/vehicle loss of control\n"
        if stats['severe_count'] > 2:
            risk += f"• ELEVATED: Multiple severe potholes increase overall road hazard\n"
        
        risk += f"\n"
        
        # Infrastructure risks
        risk += f"🗺️ **Infrastructure Risks:**\n"
        if stats['total_area'] > 5.0:
            risk += f"• Road structural integrity compromised ({stats['total_area']:.1f}m² damaged)\n"
        if stats['depth_std'] > 3:
            risk += f"• Uneven deterioration pattern indicates underlying issues\n"
        
        risk += f"\n"
        
        # Cost risks
        risk += f"💰 **Financial Risks:**\n"
        risk += f"• Delayed repairs will increase costs by 50-100% within 6 months\n"
        risk += f"• Potential liability claims if accidents occur on unrepaired sections\n"
        risk += f"• Current repair window optimal - weather/traffic conditions favorable\n"
        
        return risk
    
    def _generate_cost_estimate(self, stats):
        """Generate cost estimate in INR"""
        # Indian market rates (2024)
        MATERIAL_COST_PER_M3 = 2500  # ₹2500 per m³ of asphalt
        LABOR_COST_PER_POTHOLE = 500  # ₹500 per pothole
        EQUIPMENT_COST_BASE = 2000  # ₹2000 base
        EQUIPMENT_COST_PER_HOUR = 800  # ₹800 per hour
        
        # Estimate material with 20% overfill
        material_volume = stats['total_area'] * 0.15 * 1.2
        material_cost = material_volume * MATERIAL_COST_PER_M3
        
        # Labor cost
        labor_cost = stats['total_count'] * LABOR_COST_PER_POTHOLE
        
        # Equipment cost (estimate hours based on pothole count)
        estimated_hours = max(2, stats['total_count'] * 0.5)
        equipment_cost = EQUIPMENT_COST_BASE + (estimated_hours * EQUIPMENT_COST_PER_HOUR)
        
        # Total
        total_cost = material_cost + labor_cost + equipment_cost
        
        # Add urgency premium
        if stats['critical_count'] > 0:
            urgency_premium = total_cost * 0.25  # 25% emergency premium
            total_cost += urgency_premium
        else:
            urgency_premium = 0
        
        estimate = f"**Estimated Repair Cost (INR):**\n\n"
        estimate += f"💰 **Cost Breakdown:**\n"
        estimate += f"   • Material: ₹{material_cost:,.0f} ({material_volume:.2f}m³ @ ₹{MATERIAL_COST_PER_M3}/m³)\n"
        estimate += f"   • Labor: ₹{labor_cost:,.0f} ({stats['total_count']} potholes @ ₹{LABOR_COST_PER_POTHOLE} each)\n"
        estimate += f"   • Equipment: ₹{equipment_cost:,.0f} ({estimated_hours:.1f} hours)\n"
        
        if urgency_premium > 0:
            estimate += f"   • Emergency Premium: ₹{urgency_premium:,.0f} (25%)\n"
        
        estimate += f"\n"
        estimate += f"**TOTAL ESTIMATED COST: ₹{total_cost:,.0f}**\n\n"
        estimate += f"📊 Average cost per pothole: ₹{total_cost/stats['total_count']:,.0f}\n"
        
        # Cost comparison
        estimate += f"\n💡 **Cost Perspective:**\n"
        estimate += f"• Delaying repairs 6 months: +₹{total_cost*0.75:,.0f} (75% increase)\n"
        estimate += f"• Full road resurfacing alternative: ₹{stats['total_area']*5000:,.0f}-{stats['total_area']*8000:,.0f}\n"
        
        return estimate
    
    def _generate_no_detection_report(self):
        """Report when no potholes detected"""
        return {
            'summary': "✅ **No potholes detected in the uploaded image.**\n\nRoad surface appears to be in good condition with no visible defects.",
            'severity_report': "**Status: GOOD**\n\nNo safety concerns identified.\nRoad surface integrity maintained.",
            'recommendations': "**Maintenance Status: NORMAL**\n\n✓ Continue routine monitoring\n✓ Schedule regular inspections\n✓ No immediate action required",
            'risk_assessment': "**Risk Level: NONE**\n\n✓ No safety risks identified\n✓ Road structure appears sound\n✓ Maintain current inspection schedule",
            'cost_estimate': "**No repairs needed**\n\nCurrent maintenance cost: ₹0",
            'overall_severity': 'GOOD',
            'urgency': 'No action required',
            'risk_level': 'NONE',
            'stats': {
                'total_count': 0,
                'avg_depth': 0,
                'max_depth': 0,
                'total_area': 0,
                'critical_count': 0,
                'severe_count': 0,
                'moderate_count': 0,
                'minor_count': 0
            }
        }
    
    def generate_repair_briefing(self, detections, include_cost=True):
        """Generate complete briefing for repair teams"""
        analysis = self.analyze_detections(detections)
        
        briefing = f"""
{'='*70}
            🚨 ROAD MAINTENANCE BRIEFING
{'='*70}

{analysis['summary']}

{'─'*70}
SEVERITY ASSESSMENT
{'─'*70}
{analysis['severity_report']}

{'─'*70}
RISK ANALYSIS
{'─'*70}
{analysis['risk_assessment']}

{'─'*70}
ACTION PLAN
{'─'*70}
{analysis['recommendations']}
"""
        
        if include_cost:
            briefing += f"""
{'─'*70}
COST ESTIMATE
{'─'*70}
{analysis['cost_estimate']}
"""
        
        briefing += f"""
{'='*70}
Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Analysis ID: {self.analysis_count}
{'='*70}
"""
        
        return briefing


# ========================================
# LLM-POWERED ANALYZER (OPTIONAL)
# ========================================

class PotholeAnalyzer:
    """
    LLM-powered analyzer using Phi-2 (2.7B)
    Provides natural language insights and contextual reasoning
    Runs on CPU for stability
    """
    
    def __init__(self, model_path="./llm_models/phi-2", device="cpu"):
        """
        Initialize Phi-2 model on CPU (more stable)
        
        Args:
            model_path: Path to downloaded Phi-2 model
            device: 'cpu' recommended for stability
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("🔧 Loading Phi-2 model on CPU (stable mode)...")
            
            # Force CPU for stability
            self.device = "cpu"
            
            # Load tokenizer
            print("📦 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model on CPU (no quantization for stability)
            print("📦 Loading model weights (this may take 30-60 seconds)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # Explicitly move to CPU
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.llm_available = True
            print(f"✅ Phi-2 loaded successfully on CPU")
            print(f"💾 Model size: ~2.7B parameters")
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found at {model_path}")
            print("📥 Please download Phi-2 model first:")
            print("   git clone https://huggingface.co/microsoft/phi-2 ./llm_models/phi-2")
            self.llm_available = False
            self.fallback_analyzer = SmartPotholeAnalyzer()
            
        except Exception as e:
            print(f"⚠️ Could not load LLM: {e}")
            print("🔌 Falling back to rule-based analyzer")
            import traceback
            traceback.print_exc()
            self.llm_available = False
            self.fallback_analyzer = SmartPotholeAnalyzer()
    
    def analyze_detections(self, detections, image_size=None):
        """Generate LLM-powered analysis"""
        
        # Fallback to rule-based if LLM unavailable
        if not self.llm_available:
            return self.fallback_analyzer.analyze_detections(detections, image_size)
        
        if not detections:
            return self._generate_no_detection_report()
        
        # Compute statistics
        stats = self._compute_statistics(detections)
        
        # Build LLM prompt
        prompt = self._build_prompt(stats)
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Parse and structure
        parsed = self._parse_response(response, stats)
        
        return parsed
    
    def _compute_statistics(self, detections):
        """Extract statistics for LLM context"""
        depths = [d['depth'] for d in detections]
        areas = [d['area'] for d in detections]
        severities = [d['severity'] for d in detections]
        
        severity_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for sev in severities:
            severity_counts[sev] += 1
        
        return {
            'total_count': len(detections),
            'avg_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'min_depth': np.min(depths),
            'avg_area': np.mean(areas),
            'total_area': np.sum(areas),
            'severity_distribution': severity_counts,
            'critical_count': severity_counts[4],
            'severe_count': severity_counts[3],
            'moderate_count': severity_counts[2],
            'minor_count': severity_counts[1]
        }
    
    def _build_prompt(self, stats):
        """Build contextual prompt for LLM"""
        prompt = f"""You are a road infrastructure expert providing natural language analysis.

Detection Results:
- Total potholes: {stats['total_count']}
- Average depth: {stats['avg_depth']:.1f}cm (max: {stats['max_depth']:.1f}cm)
- Total damaged area: {stats['total_area']:.2f}m²
- Critical: {stats['critical_count']}, Severe: {stats['severe_count']}, Moderate: {stats['moderate_count']}, Minor: {stats['minor_count']}

Provide a conversational analysis covering:
1. Overall road condition with context
2. Why specific potholes are concerning
3. Strategic repair recommendations
4. Cost-benefit considerations

Analysis:"""
        
        return prompt
    
    def _generate_response(self, prompt, max_new_tokens=256):
        """Generate LLM response with safety checks"""
        import torch
        
        try:
            # Tokenize with proper padding and attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            print("🤖 Generating LLM response...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            generated = response[len(prompt):].strip()
            
            # Fallback if generation is empty
            if not generated or len(generated) < 10:
                print("⚠️ LLM generated empty response, using fallback")
                return "Road analysis completed. Check detailed statistics above."
            
            print(f"✅ Generated {len(generated)} characters")
            return generated
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return "Analysis completed. See statistics above for details."
    
    def _parse_response(self, response, stats):
        """Structure LLM response"""
        
        # Determine severity
        if stats['critical_count'] > 0:
            overall_severity = "CRITICAL"
            urgency = "Immediate (within 24 hours)"
            risk_level = "EXTREME"
        elif stats['severe_count'] > 2:
            overall_severity = "SEVERE"
            urgency = "High (within 2-3 days)"
            risk_level = "HIGH"
        else:
            overall_severity = "MODERATE"
            urgency = "Medium (within 1-2 weeks)"
            risk_level = "MODERATE"
        
        # Use LLM response as main content
        summary = response[:500] if len(response) > 500 else response
        
        return {
            'summary': summary,
            'severity_report': f"**{risk_level} Risk**\n\n{response[500:1000] if len(response) > 500 else 'See summary above'}",
            'recommendations': f"**{urgency}**\n\n{response[1000:] if len(response) > 1000 else 'Strategic repairs recommended based on analysis above'}",
            'raw_llm_response': response,
            'overall_severity': overall_severity,
            'risk_level': risk_level,
            'urgency': urgency,
            'stats': stats
        }
    
    def _generate_no_detection_report(self):
        """No detections response"""
        return {
            'summary': "✅ No potholes detected. Road surface appears to be in good condition.",
            'severity_report': "**Status: GOOD** - No safety concerns identified.",
            'recommendations': "Continue routine monitoring and inspections.",
            'overall_severity': 'GOOD',
            'urgency': 'No action required',
            'risk_level': 'NONE',
            'stats': {'total_count': 0}
        }


# Backward compatibility
SimplePotholeAnalyzer = SmartPotholeAnalyzer
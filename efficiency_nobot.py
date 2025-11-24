import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns

class NobotAnalyzer:
    """Analyze and visualize nobot performance data"""
    
    def __init__(self, json_file: str = "nobot_performance_detailed.json"):
        """Load performance data from JSON file"""
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.functions = self.data.get('functions', {})
        self.summary = self.data.get('summary', {})
    
    def create_performance_dashboard(self, save_path: str = "nobot_dashboard.png"):
        """Create comprehensive performance dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Nobot Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Extract function names and metrics
        func_names = list(self.functions.keys())
        
        # 1. Execution Time Comparison
        exec_times = []
        for func in func_names:
            time_str = self.functions[func]['execution_time']['mean']
            exec_times.append(float(time_str.replace('s', '')))
        
        axes[0, 0].bar(func_names, exec_times, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[0, 0].set_title('Average Execution Time', fontweight='bold')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Throughput Comparison
        throughputs = []
        for func in func_names:
            throughput_str = self.functions[func]['throughput']
            throughputs.append(float(throughput_str.split()[0]))
        
        axes[0, 1].bar(func_names, throughputs, color=['#9b59b6', '#f39c12', '#1abc9c'])
        axes[0, 1].set_title('Throughput', fontweight='bold')
        axes[0, 1].set_ylabel('Calls per Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Success Rate
        success_rates = []
        for func in func_names:
            rate_str = self.functions[func]['success_rate']
            success_rates.append(float(rate_str.replace('%', '')))
        
        colors_success = ['#27ae60' if r == 100 else '#e67e22' for r in success_rates]
        axes[1, 0].bar(func_names, success_rates, color=colors_success)
        axes[1, 0].set_title('Success Rate', fontweight='bold')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_ylim([0, 105])
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=100, color='green', linestyle='--', alpha=0.3)
        
        # 4. Performance Summary Table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = []
        for func in func_names:
            f_data = self.functions[func]
            table_data.append([
                func[:20],  # Truncate long names
                f_data['total_calls'],
                f_data['execution_time']['mean'],
                f_data['throughput']
            ])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Function', 'Calls', 'Avg Time', 'Throughput'],
            cellLoc='left',
            loc='center',
            colWidths=[0.4, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[1, 1].set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to {save_path}")
        plt.close()
    
    def create_latency_distribution(self, save_path: str = "latency_distribution.png"):
        """Create latency distribution chart with percentiles"""
        if not any('detailed_measurements' in f for f in self.functions.values()):
            print("⚠️  No detailed measurements available. Run profiler with detailed=True")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for func_name, func_data in self.functions.items():
            if 'detailed_measurements' in func_data:
                times = [float(m['execution_time'].replace('s', '')) 
                        for m in func_data['detailed_measurements']]
                
                ax.hist(times, bins=20, alpha=0.6, label=func_name, edgecolor='black')
        
        ax.set_xlabel('Execution Time (seconds)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Latency Distribution Across Functions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Latency distribution saved to {save_path}")
        plt.close()
    
    def create_timeline_chart(self, save_path: str = "performance_timeline.png"):
        """Create performance over time chart"""
        if not any('detailed_measurements' in f for f in self.functions.values()):
            print("⚠️  No detailed measurements available. Run profiler with detailed=True")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for func_name, func_data in self.functions.items():
            if 'detailed_measurements' in func_data:
                measurements = func_data['detailed_measurements']
                indices = list(range(len(measurements)))
                times = [float(m['execution_time'].replace('s', '')) * 1000  # Convert to ms
                        for m in measurements]
                
                ax.plot(indices, times, marker='o', markersize=3, 
                       label=func_name, linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Call Number', fontweight='bold')
        ax.set_ylabel('Execution Time (milliseconds)', fontweight='bold')
        ax.set_title('Performance Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Timeline chart saved to {save_path}")
        plt.close()
    
    def export_to_csv(self, save_path: str = "nobot_performance.csv"):
        """Export performance data to CSV for Excel analysis"""
        rows = []
        
        for func_name, func_data in self.functions.items():
            if 'detailed_measurements' in func_data:
                for measurement in func_data['detailed_measurements']:
                    rows.append({
                        'function': func_name,
                        'timestamp': measurement['timestamp'],
                        'execution_time_s': float(measurement['execution_time'].replace('s', '')),
                        'memory_mb': float(measurement['memory_mb']),
                        'cpu_percent': float(measurement['cpu_percent'])
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(save_path, index=False)
            print(f"✅ CSV exported to {save_path}")
            print(f"   Total records: {len(rows)}")
        else:
            print("⚠️  No detailed measurements to export")
    
    def generate_html_report(self, save_path: str = "nobot_report.html"):
        """Generate interactive HTML report"""
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Nobot Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .summary-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .function-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .danger {{ color: #e74c3c; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Nobot Performance Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="summary-box">
        <h2>📊 Summary</h2>
        {summary_metrics}
    </div>
    
    {function_cards}
    
    <div class="summary-box">
        <h2>💡 Recommendations</h2>
        <ul>
            <li>Monitor P95/P99 latencies under production load</li>
            <li>Implement caching for frequently verified patterns</li>
            <li>Set up real-time monitoring with alerts</li>
            <li>Load test with 10x expected traffic</li>
        </ul>
    </div>
</body>
</html>
"""
        
        # Build summary metrics HTML
        summary_html = ""
        for key, value in self.summary.items():
            summary_html += f'<div class="metric"><div class="metric-label">{key.replace("_", " ").title()}</div><div class="metric-value">{value}</div></div>'
        
        # Build function cards HTML
        function_cards_html = ""
        for func_name, func_data in self.functions.items():
            if 'error' in func_data:
                continue
            
            success_rate = float(func_data['success_rate'].replace('%', ''))
            success_class = 'success' if success_rate == 100 else 'warning' if success_rate > 90 else 'danger'
            
            card_html = f"""
    <div class="function-card">
        <h3>⚙️ {func_name}</h3>
        <div class="metric">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value {success_class}">{func_data['success_rate']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Throughput</div>
            <div class="metric-value">{func_data['throughput']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Calls</div>
            <div class="metric-value">{func_data['total_calls']}</div>
        </div>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>P95</th>
                <th>P99</th>
                <th>Max</th>
            </tr>
            <tr>
                <td>Execution Time</td>
                <td>{func_data['execution_time']['mean']}</td>
                <td>{func_data['execution_time']['p95']}</td>
                <td>{func_data['execution_time']['p99']}</td>
                <td>{func_data['execution_time']['max']}</td>
            </tr>
        </table>
    </div>
"""
            function_cards_html += card_html
        
        # Generate final HTML
        html_content = html_template.format(
            timestamp=self.data['timestamp'],
            summary_metrics=summary_html,
            function_cards=function_cards_html
        )
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ HTML report saved to {save_path}")
        print(f"   Open it in your browser to view the interactive report")
    
    def generate_all_reports(self):
        """Generate all visualization and export formats"""
        print("\n🎨 Generating comprehensive analysis reports...\n")
        
        self.create_performance_dashboard()
        self.create_latency_distribution()
        self.create_timeline_chart()
        self.export_to_csv()
        self.generate_html_report()
        
        print("\n✅ All reports generated successfully!")
        print("📁 Files created:")
        print("   • nobot_dashboard.png - Visual performance overview")
        print("   • latency_distribution.png - Response time distribution")
        print("   • performance_timeline.png - Performance over time")
        print("   • nobot_performance.csv - Raw data for Excel")
        print("   • nobot_report.html - Interactive HTML report")


# Usage example
if __name__ == "__main__":
    # Analyze the JSON file generated by the profiler
    analyzer = NobotAnalyzer("nobot_performance_detailed.json")
    
    # Generate all reports at once
    analyzer.generate_all_reports()
    
    # Or generate individual reports
    # analyzer.create_performance_dashboard()
    # analyzer.export_to_csv()
    # analyzer.generate_html_report()
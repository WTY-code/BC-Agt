import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def parse_log_file(file_path):
    """Parse a log file and extract CPU and memory metrics."""
    data = []
    # Pattern to match the monitoring data in the logs
    pattern = r'\[(.*?)\] CPU: ([\d.]+)%% Memory: Total ([\d.]+)MB, Used ([\d.]+)MB, Available ([\d.]+)MB, Usage ([\d.]+)%'
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                    cpu_percent = float(match.group(2))
                    memory_used_mb = float(match.group(4))
                    memory_used_gb = memory_used_mb / 1024  # Convert MB to GB
                    memory_usage_percent = float(match.group(6))
                    
                    data.append({
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent,
                        'memory_used_gb': memory_used_gb,
                        'memory_usage_percent': memory_usage_percent,
                        'source': os.path.basename(file_path)  # Extract filename for the legend
                    })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()

def main():
    # List of log files to process
    log_files = [
        './scripts/orderer.example.com_monitor.log',
        './scripts/peer0.org1.example.com_monitor.log',
        './scripts/peer0.org2.example.com_monitor.log'
    ]
    
    # Parse and combine all data
    all_data = pd.DataFrame()
    for file_path in log_files:
        df = parse_log_file(file_path)
        if not df.empty:
            all_data = pd.concat([all_data, df])
            print(f"Successfully parsed {file_path}, found {len(df)} data points.")
    
    # If we have data, sort it and create the plots
    if not all_data.empty:
        all_data = all_data.sort_values('timestamp')
        
        # Create the plots
        plt.figure(figsize=(15, 12))
        
        # Plot CPU over time
        plt.subplot(3, 1, 1)
        for source in all_data['source'].unique():
            source_data = all_data[all_data['source'] == source]
            plt.plot(source_data['timestamp'], source_data['cpu_percent'], label=source)
        plt.title('CPU Usage Over Time')
        plt.ylabel('CPU Usage (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Memory Used (GB) over time
        plt.subplot(3, 1, 2)
        for source in all_data['source'].unique():
            source_data = all_data[all_data['source'] == source]
            plt.plot(source_data['timestamp'], source_data['memory_used_gb'], label=source)
        plt.title('Memory Used Over Time (GB)')
        plt.ylabel('Memory Used (GB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Memory Usage Percentage over time
        plt.subplot(3, 1, 3)
        for source in all_data['source'].unique():
            source_data = all_data[all_data['source'] == source]
            plt.plot(source_data['timestamp'], source_data['memory_usage_percent'], label=source)
        plt.title('Memory Usage Percentage Over Time')
        plt.ylabel('Memory Usage (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('./scripts/system_monitoring.png')
        plt.show()
        
        print("Plots created successfully and saved as 'system_monitoring.png'!")
    else:
        print("No data was found in the log files.")

if __name__ == "__main__":
    main()
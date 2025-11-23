# HRV Project

## Overview
The HRV Project is designed to analyze heart rate variability (HRV) metrics through various computational methods and provide a dashboard for visualizing the results. This project aims to facilitate research and applications in health monitoring and analysis.

## Installation
To install the necessary dependencies, please run:

```
pip install -r requirements.txt
```

## Usage
1. **HRV Metrics**: Use the functions in `src/hrv_metrics/metrics.py` to calculate various HRV statistics.
2. **Data Streaming**: The `src/streaming/consumer.py` file handles data streaming from specified sources.
3. **Dashboard**: Launch the application using the main logic defined in `src/dashboard/app.py`.

## Directory Structure
- `src/`: Contains the main source code for the project.
  - `hrv_metrics/`: Functions and classes related to HRV metrics.
  - `streaming/`: Handles data streaming functionalities.
  - `dashboard/`: Contains the dashboard application logic.
- `docs/`: Documentation files, including architecture details.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `data/`: Directory for storing datasets used in the project.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

The Harrington dataset represents a multi-response optimization problem with 3
independent variables (X1, X2, X3) and 5 dependent variables (V1-V5). The overall desirability (D)
is calculated by Harrington as described in the original article.[1] The dataset is extracted from the


| Variable ID | Variable Name          | Units    | Min  | Max  |
|-------------|------------------------|----------|------|------|
| X1          | Temperature       	    | °C       |      |      |
| X2          | Catalyst               | %        |      | None |
| X3          | Modifier               | %        |      |      |
| V1          | Tensile Strength       | psi      | 7000 | None |
| V2          | Tensile Modulus        | 10^5 psi | 4.5  | 5.5  |
| V3          | Deflection Temperature | °F       | 190  | None |
| V4          | Dielectric Constant    |          | None | 2.6  |
| V5          | Flow Index             |          | 1.20 | 1.40 |
| D           | Overall Desirability   |          | 1.20 | 1.40 |


harrington_dataset

| index | X1    | X2    | X3    | V1     | V2     | V3     | V4   | V5   | Dh   |
|-------|-------|-------|-------|--------|--------|--------|------|------|------|
| 1     | 50    | 0.02  | 0.00  | 8381   | 4.29   | 187.60 | 2.48 | 1.54 | 0.00 |
| 2     | 60    | 0.02  | 0.00  | 7200   | 5.06   | 191.1  | 2.50 | 1.31 | 0.74 |
| 3     | 55    | 0.02  | 0.02  | 9927   | 4.58   | 190.4  | 2.52 | 1.41 | 0.44 |
| 4     | 50    | 0.02  | 0.04  | 11611  | 4.13   | 190.1  | 2.54 | 1.50 | 0.00 |
| 5     | 60    | 0.02  | 0.04  | 8380   | 5.35   | 193.6  | 2.63 | 1.23 | 0.64 |
| 6     | 55    | 0.03  | 0.00  | 6947   | 4.44   | 190.2  | 2.41 | 1.41 | 0.34 |
| 7     | 50    | 0.03  | 0.02  | 8927   | 4.00   | 187.20 | 2.47 | 1.47 | 0.00 |
| 8     | 55    | 0.03  | 0.02  | 8453   | 4.47   | 190.1  | 2.50 | 1.38 | 0.56 |
| 9     | 60    | 0.03  | 0.02  | 6947   | 5.11   | 191.2  | 2.57 | 1.24 | 0.65 |
| 10    | 55    | 0.03  | 0.04  | 8927   | 4.58   | 192.20 | 2.58 | 1.35 | 0.70 |
| 11    | 50    | 0.04  | 0.00  | 6600   | 3.95   | 187.20 | 2.42 | 1.46 | 0.00 |
| 12    | 60    | 0.04  | 0.00  | 5870   | 4.97   | 191.7  | 2.54 | 1.26 | 0.53 |
| 13    | 55    | 0.04  | 0.02  | 8147   | 4.44   | 190.0  | 2.57 | 1.35 | 0.50 |
| 14    | 50    | 0.04  | 0.04  | 9380   | 3.94   | 188.70 | 2.61 | 1.42 | 0.06 |
| 15    | 60    | 0.04  | 0.04  | 6600   | 5.41   | 193.2  | 2.80 | 1.18 | 0.08 |
| 79    | 47.50 | 0.02  | 0.03  | 9540   | 490.00 | 192.00 | 2.56 | 1.33 | 0.82 |


[1] Harrington, E. C. (1965). "The Desirability Function." Industrial Quality Control(21): 494-498.

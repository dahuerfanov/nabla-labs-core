# Perfect Fit Templates Adapters

This directory contains adapters that bridge the **product parsing** domain (Perfect Fit submodule) and the **garment generation** domain (Garment Code).

## Available Adapters

- **`PantsAdapter`**: Converts `PantsSchema` → `PerfectFitPantsDesign` parameters
- **`ShirtAdapter`**: Converts `UpperSchema` → `PerfectFitShirtDesign` parameters

## Architecture

Each adapter follows a straightforward transformation pattern:

1.  **Input**: Schema from `perfect_fit.product_parsing.schemas.*`
    - Represents structured data extracted from image analysis
2.  **Processing**: Adapter class with mapping methods
    - Contains precise mapping logic to translate high-level descriptions/percentages into parametric design values
3.  **Output**: `dict` of parameters
    - A dictionary ready to be unpacked into the corresponding `PerfectFit*Design` constructor

> **Note**: Material properties (`garment_thickness`, `fabric_stretch`) are currently ignored and will emit a logger warning.

---

## PantsAdapter

### Property Mapping Logic

#### 1. Rise Height
Maps a percentage value (0-150) to a `RaiseWaist` enum.
- **≤ 70**: `RaiseWaist.LOW`
- **71 - 100**: `RaiseWaist.MID`
- **> 100**: `RaiseWaist.HIGH`

#### 2. Crotch Displacement
Maps `CrotchDisplacement` string enums to `CrotchDisplacementRatio` numeric enums.
- *Above Anatomical* → `INSIDE_CROTCH`
- *At Anatomical* → `AT_CROTCH`
- *Slightly Below* → `LOW_CROTCH`
- *Significantly Below/Extreme* → `HANGY`

#### 3. General Fit (Upper Leg)
Maps `UpperLeg` descriptions to `GeneralFit` ratios.
- *Super Tight* → `SUPER_TIGHT` (0.8)
- *Tight* → `TIGHT` (0.9)
- *Regular/Straight* → `REGULAR` (1.0)
- *Wide/Baggy* → `WIDE` (1.1)

#### 4. Leg Length
Maps length percentage to `LegLength` enum.
- **< 20**: `SHORT_SHORTS`
- **20 - 54**: `SHORTS`
- **55 - 94**: `CAPRI`
- **95 - 100**: `ANKLE`
- **100 - 109**: `FULL`
- **≥ 110**: `LONG`

#### 5. Pants Style Spec
Combines `UpperLeg` and `LegOpening` to determine the overall silhouette.
- **Tights**: If upper leg is super tight
- **Skinny**: If upper leg is tight AND leg opening is tapered/na
- **Flare/Bootcut**: If leg opening is explicitly flare or bootcut
- **Wide Leg**: If leg opening is wide
- **Straight**: Default for other combinations

#### 6. Cuff
Maps `CuffType` to a specific `cuff_length_ratio`.
- *No Cuff*: 0.0
- *Short*: 0.02
- *Medium*: 0.05
- *Long*: 0.08

### Usage Example

```python
from perfect_fit.product_parsing.schemas.pants_schema import PantsSchema
from perfect_fit_templates.adapter import PantsAdapter

schema = PantsSchema(
    pant_length=100,
    rise_height=80,
    upper_leg="straight",
    leg_opening="straight_opening",
    cuff_type="short_cuff",
    crotch_displacement="at_anatomical_crotch",
    garment_thickness="normal",
    fabric_stretch="medium_stretch"
)

adapter = PantsAdapter()
design_params = adapter.to_garment_code_design(schema)
```

---

## ShirtAdapter

The perfect-fit design parameters (along with their names) are based on the [garment design document](https://drive.google.com/drive/folders/1USQhfuj0dCg2Zwl_K7wlXr4T36EeBHE7).

### Property Mapping Logic

#### 1. Torso Length → Front/Back Length Ratios
Maps percentage (0-125) to length ratios.
- **0-70**: Ultra-cropped (0.65-0.8)
- **71-90**: Waist length (0.8-0.95)
- **91-100**: Hip length (0.95-1.0)
- **101-110**: Lower hip (1.0-1.1)
- **111-125**: Mid-thigh (1.1-1.25)

#### 2. Torso Shape → Width Ratios
| Shape | Chest | Waist | Hip |
|-------|-------|-------|-----|
| TIGHT_BODYCON | 0.9 | 0.85 | 0.9 |
| REGULAR_STRAIGHT_FIT | 1.0 | 1.0 | 1.0 |
| BOX_CROPPED_FIT | 1.05 | 1.05 | 1.0 |
| A_LINE_FLARE | 1.0 | 1.1 | 1.2 |
| OVERSIZED_BAGGY | 1.2 | 1.2 | 1.2 |
| SHIRT_BUTTONUP_FIT | 1.0 | 0.95 | 1.0 |

#### 3. Neckline → Neck Width Ratio
| Neckline | Ratio |
|----------|-------|
| THIN_STRAPS/THICK_STRAPS | 0.3 |
| SCOOP_OVAL_NECK | 0.4 |
| V_NECK | 0.35 |
| CREW_ROUND_NECK | 0.25 |
| TURTLENECK | 0.2 |
| HOOD/COLLARED | 0.3 |

#### 4. Sleeve Length → Sleeve Length Ratio
- **0-10**: Sleeveless (sets `sleeveless=True`)
- **11-50**: Short (0.3-0.5)
- **51-75**: 3/4 (0.5-0.75)
- **76-100**: Full (0.75-1.0)
- **101-115**: Extra long (1.0-1.15)

#### 5. Sleeve Fit → Arm Width Ratios
*Note: Ratios are relative to arm length, not arm circumference.*

| Fit | Bicep | Elbow | Wrist |
|-----|-------|-------|-------|
| TIGHT_FITTED | 0.22 | 0.20 | 0.14 |
| REGULAR_STRAIGHT | 0.26 | 0.24 | 0.16 |
| WIDE_LOOSE | 0.29 | 0.26 | 0.18 |
| VOLUMINOUS_PUFF | 0.30 | 0.26 | 0.14 |
| INTEGRATED_KIMONO | 0.28 | 0.26 | 0.18 |

#### 6. Sleeve Cuff → Cuff Length
- *No Cuff*: 0.0
- *Ribbed Band*: 0.03
- *Shirt Cuff Buttoned*: 0.05
- *Drawstring*: 0.02
- *Flare/Bell*: 0.0

### Default Values

The following parameters use global defaults since they cannot be mapped from the schema:
- `smallest_width_above_chest_ratio`: 0.9
- `neck_to_shoulder_distance_ratio`: 0.5
- `shoulder_slant_ratio`: 0.15
- `waist_over_bust_line_height_ratio`: 0.7
- `back_width_ratio`: 0.95
- `scye_depth_ratio`: 0.6
- `armhole_size_ratio`: 0.25

### Usage Example

```python
from perfect_fit.product_parsing.schemas.upper_schema import UpperSchema
from perfect_fit_templates.shirt_adapter import ShirtAdapter

schema = UpperSchema(
    torso_length=100,
    torso_neckline_type="crew_round_neck",
    torso_garment_shape="regular_straight_fit",
    torso_hem_finish="plain_hem",
    torso_hem_width="regular_straight",
    sleeve_length=80,
    sleeve_fit="regular_straight",
    sleeve_cuff_type="no_cuff_plain_hem",
    garment_thickness="normal",
    fabric_stretch="medium_stretch"
)

adapter = ShirtAdapter()
design_params = adapter.to_garment_code_design(schema)
```

---

## Validation Rules

To ensure valid garment geometry, the following validation rules are enforced during `PerfectFitShirtDesign` initialization:

### 1. Width Consistency (`WidthConsistencyRule`)
Ensures narrower body parts do not exceed wider parts (relative to their specific anatomical ratios).
*   `smallest_width_above_chest_ratio` ≤ `width_chest_ratio`
*   `back_width_ratio` ≤ `width_chest_ratio`

### 2. Length Proportions (`LengthProportionRule`)
Ensures vertical anatomical landmarks remain in the correct order.
*   `scye_depth_ratio` < `waist_over_bust_line_height_ratio`
*   `waist_over_bust_line_height_ratio` < `front_length_ratio`
*   `waist_over_bust_line_height_ratio` < `back_length_ratio`

### 3. Sleeve Geometry (`SleeveGeometryRule`)
Ensures the sleeve cap width fits within the armhole opening to prevent geometric errors.
*   `bicep_width` (calculated absolute) < `armhole_height` (approx `scye` - `slant`)

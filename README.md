# SVRIMG-47: Detailed Convective Morphology Samples from NEXRAD Mosaics

A complementary effort to the original [SVRIMG](http://www.svrimg.org/) by Alex Haberlie (LSU) and collaborators for detailed storm classification.

## Summary

This repository contains the work-in-progress code powering the generation of the SVRIMG-47 dataset and sample images as well as the labeling web application. Data analysis utilties are planned for inclusion in the near future, as is more detailed description in this repository and accompanying website.

For a prelimary description of the dataset and the philosophy behind it, see [the informal dataset proposal here](https://hackmd.io/@jthielen/morphology_dataset).

## Source Datasets

SVRIMG-47 consists of data reprocessed from a variety of source NEXRAD-derived data products:

- [MYRORSS](https://osf.io/8f4v2/) (for 1998-2011 only)
- GridRad-Severe (2011 only to begin, will fill in upon public release)
- Archived MRMS (excluded to begin, will fill in upon archived product contributions)
- Custom mosiacs using [OpenMosaic](https://github.com/jthielen/OpenMosaic) (full 1998-2019 period)

All products are regridded from their original grids to a uniform 2km Lambert Conformal grid (except for OpenMosaic products, which directly grid the Level II data onto this target grid). CF attributes for this grid mapping are:

```python
cf_attrs = {
    'grid_mapping_name': 'lambert_conformal_conic',
    'latitude_of_projection_origin': 38.5,
    'longitude_of_central_meridian': 262.5,
    'standard_parallel': 38.5,
    'earth_radius': 6371229.0
}
```

Individual samples are then extracted based on severe weather reports (as well as currently-under-development non-severe storm sampling methods).

## License Information

Copyright 2021 Jonathan Thielen.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


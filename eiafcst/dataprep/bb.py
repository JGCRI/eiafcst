# Define bounding box constants for spatial regions.
#
# NOTE:
# The bounding boxes defined here for Alaska and Hawaii do not include each
# entire state! The selections include the limited areas for which we have
# electricity data. For Alaska this is around Anchorage, and for Hawaii this
# is around Honolulu.
#
# Bounding boxes should be in integer degrees, as the CDS data is at a
# resolution of 0.25, 0.5, or 1 degree, depending on the variable and request.
#
# Caleb Braun
# 1/28/19

# Bounding boxes for the continental US, AK, and HI
# [x1, y1, x2, y2]

# CONTINENTAL_BB = [-125.0011, 24.9493, -66.9326, 49.5904]
CONTINENTAL_BB = [-126, 25, -67, 50]

# AK_BB = [-155.0, 58.0, -147.5, 62.0]
AK_BB = [-155, 58, -148, 62]

# HI_BB = [-158.3, 21.2, -157.6, 21.8]
HI_BB = [-159, 21, -157, 22]

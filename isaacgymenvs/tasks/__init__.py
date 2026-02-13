# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .franka_leap import FrankaLEAP
from .franka_leap_pick_table import FrankaLEAPPickTable
from .franka_leap_pick_table_side import FrankaLEAPPickTableSide
from .franka_leap_pick_top import FrankaLEAPPickTop
from .franka_leap_pick_side import FrankaLEAPPickSide
from .franka_leap_pick_full import FrankaLEAPPickFull
from .franka_leap_pick_top_full import FrankaLEAPPickTopFull

from .franka_leap_mobile import FrankaLEAPMobile
from .franka_leap_mobile_pick_table import FrankaLEAPMobilePickTable
from .franka_leap_mobile_pick_table_multi import FrankaLEAPMobilePickTableMulti
from .franka_leap_mobile_pick_top_full import FrankaLEAPMobilePickTopFull

from .franka_cmd import FrankaCMD
from .franka_cmd_pick_table import FrankaCMDPickTable



# Mappings from strings to environments
isaacgym_task_map = {
    "FrankaLEAP": FrankaLEAP,
    "FrankaLEAPPickTable": FrankaLEAPPickTable,
    "FrankaLEAPPickTableSide": FrankaLEAPPickTableSide,
    "FrankaLEAPPickTop": FrankaLEAPPickTop,
    "FrankaLEAPPickSide": FrankaLEAPPickSide,
    "FrankaLEAPPickFull": FrankaLEAPPickFull,
    "FrankaLEAPPickTopFull": FrankaLEAPPickTopFull,
    "FrankaLEAPMobile": FrankaLEAPMobile,
    "FrankaLEAPMobilePickTable": FrankaLEAPMobilePickTable,
    "FrankaLEAPMobilePickTableMulti": FrankaLEAPMobilePickTableMulti,
    "FrankaLEAPMobilePickTopFull": FrankaLEAPMobilePickTopFull,
    "FrankaCMD": FrankaCMD,
    "FrankaCMDPickTable": FrankaCMDPickTable,
}

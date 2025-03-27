#pragma once
/*
 *  Copyright (C) 2024  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef IMAGES_H
#define IMAGES_H

#include <nn/config.h>
#include <string>
#include <vector>

namespace nn
{
    struct image_row_t
    {
        std::string path;
        std::string original_path;
        blt::u32 x, y, width, height;
    };

    struct image_data_t
    {
    };

    std::vector<image_row_t> load_image_list(const std::string& path);
}

#endif //IMAGES_H

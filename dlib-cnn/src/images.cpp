/*
 *  <Short Description>
 *  Copyright (C) 2025  Brett Terpstra
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
#include <nn/images.h>
#include <blt/fs/loader.h>

namespace nn
{
    std::vector<image_row_t> load_image_list(const std::string& path)
    {
        std::vector<image_row_t> rows;
        auto lines = blt::fs::getLinesFromFile(path);

        for (const auto& line : lines)
        {
            auto parts = blt::string::split(line, ',');
            image_row_t row;
            row.path = parts[0];
            row.original_path = parts[1];
            row.x = std::stoi(parts[2]);
            row.y = std::stoi(parts[3]);
            row.width = std::stoi(parts[4]);
            row.height = std::stoi(parts[5]);
            rows.push_back(row);
        }

        return rows;
    }
}

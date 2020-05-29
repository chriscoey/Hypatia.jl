function read_instances(setfile::String)
    instances = SubString[]
    for l in readlines(setfile)
        str = split(strip(l))
        if !isempty(str)
            str1 = first(str)
            if !startswith(str1, '#')
                push!(instances, str1)
            end
        end
    end
    return instances
end

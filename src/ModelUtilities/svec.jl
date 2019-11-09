#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

function vec_to_svec!(vec::AbstractVector{T}, rt2::T) where {T}
    side = round(Int, sqrt(0.25 + 2 * length(vec)) - 0.5)
    k = 0
    for i in 1:side
        for j in 1:(i - 1)
            k += 1
            vec[k] *= rt2
        end
        k += 1
    end
    return vec
end

function vec_to_svec_cols!(A::Matrix{T}, rt2::T) where {T}
    @views for j in 1:size(A, 2)
        vec_to_svec!(A[:, j], rt2)
    end
    return A
end

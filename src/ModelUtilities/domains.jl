#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

definitions of Hypatia domains
=#

abstract type Domain{T <: Real} end

# hyperrectangle/box
mutable struct Box{T <: Real} <: Domain{T}
    l::Vector{T}
    u::Vector{T}
    function Box{T}(l::Vector{T}, u::Vector{T}) where {T <: Real}
        @assert length(l) == length(u)
        dom = new{T}()
        dom.l = l
        dom.u = u
        return dom
    end
end

get_dimension(dom::Box) = length(dom.l)
get_degree(::Box) = 2

# Euclidean hyperball
mutable struct Ball{T <: Real} <: Domain{T}
    c::Vector{Float64}
    r::Float64
    function Ball{T}(c::Vector{T}, r::T) where {T <: Real}
        dom = new{T}()
        dom.c = c
        dom.r = r
        return dom
    end
end

get_dimension(dom::Ball) = length(dom.c)
get_degree(::Ball) = 2

# hyperellipse: (x-c)'Q(x-c) \leq 1
mutable struct Ellipsoid{T <: Real} <: Domain{T}
    c::Vector{Float64}
    Q::AbstractMatrix{Float64}
    function Ellipsoid{T}(c::Vector{T}, Q::AbstractMatrix{T}) where {T <: Real}
        @assert length(c) == size(Q, 1)
        dom = new{T}()
        dom.c = c
        dom.Q = Q
        return dom
    end
end

get_dimension(dom::Ellipsoid) = length(dom.c)
get_degree(::Ellipsoid) = 2

# All reals of dimension n
mutable struct FreeDomain{T <: Real} <: Domain{T}
    n::Int
end

get_dimension(dom::FreeDomain) = dom.n
get_degree(::FreeDomain) = 0

# TODO replace this domain with a more general cartesian product
# assumes the free part has the same dimension as the restricted part
mutable struct SemiFreeDomain{T <: Real} <: Domain{T}
    restricted_halfregion::Domain{T}
end

get_dimension(dom::SemiFreeDomain) = 2 * get_dimension(dom.restricted_halfregion)
get_degree(dom::SemiFreeDomain) = get_degree(dom.restricted_halfregion)

add_free_vars(dom::Domain) = SemiFreeDomain(dom)

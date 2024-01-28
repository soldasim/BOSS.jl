
# - - - Expected - - - - -

abstract type Expected end

struct Success <: Expected
    assert
end
Success() = Success((in, out) -> true)

struct Failure <: Expected
    exception::Type

    function Failure(exception)
        @assert exception <: Exception
        new(exception)
    end
end
Failure() = Failure(Exception)


# - - - Macros - - - - -

macro params(expr)
    if (expr isa Expr) && (expr.head == :tuple)
        return expr |> esc
    else
        return Expr(:tuple, expr) |> esc
    end
end

is_params_call(expr) = (expr.head == :macrocall) && (expr.args[1] == Symbol("@params"))

"""
# Examples

`@success out == 1`

`@success in[1] + in[2] == out`
"""
macro success()
    return Expr(:call, :Success) |> esc
end
macro success(expr)
    body = (expr.head == :tuple) ?
        Expr(:call, :all, Expr(:vect, expr.args...)) :
        expr
    assert = Expr(:->, Expr(:tuple, :in, :out), body)
    return Expr(:call, :Success, assert) |> esc
end

is_success_call(expr) = (expr.head == :macrocall) && (expr.args[1] == Symbol("@success"))

"""
# Examples

`@failure BoundsError`
"""
macro failure()
    return Expr(:call, :Failure) |> esc
end
macro failure(expr)
    return Expr(:call, :Failure, expr) |> esc
end

is_failure_call(expr) = (expr.head == :macrocall) && (expr.args[1] == Symbol("@failure"))

macro param_test(unit, expr)
    Base.remove_linenums!(expr)
    assert_param_test_structure(expr)

    params = Expr[]
    expected = Expr[]
    for e in expr.args
        if is_params_call(e)
            push!(params, e)
        else
            while length(expected) < length(params)
                push!(expected, e)
            end
        end
    end

    params_var = gensym("params")
    expected_var = gensym("expected")

    return Expr(:block,
        Expr(:(=), params_var, Expr(:vect, params...)),
        Expr(:(=), expected_var, Expr(:vect, expected...)),
        Expr(:call, :parametrized_test, unit, params_var, expected_var)
    ) |> esc
end

function assert_param_test_structure(expr)
    @assert all((is_params_call(e) || is_success_call(e) || is_failure_call(e) for e in expr.args))
    @assert is_params_call(expr.args[1])
    @assert is_success_call(expr.args[end]) || is_failure_call(expr.args[end])
end


# - - - Scripts - - - - -

parametrized_test(unit, params::AbstractVector{<:Tuple}, expected::AbstractVector{<:Expected}) =
    parametrized_test.(eachindex(expected), Ref(unit), params, expected)

parametrized_test(idx, script, inputs, exp::Success) = @testset "$idx" begin @test exp.assert(inputs, script(inputs...)) end
parametrized_test(idx, script, inputs, exp::Failure) = @testset "$idx" begin @test_throws exp.exception script(inputs...) end

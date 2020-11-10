import simpy


def resource_user(env, resource):
    while True:
        request = resource.request()  # Generate a request event
        print('request generated')
        yield request
        print('request start being served') # Wait for access
        yield env.timeout(1)          # Do something
        print('request leaves the station')
        resource.release(request)     # Release the resource

env = simpy.Environment()
res = simpy.Resource(env, capacity=5)
user = env.process(resource_user(env, res))
env.run(100)











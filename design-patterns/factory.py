import abc
import urllib2

from BeautifulSoup import BeautifulSoup as BS


# Factory method pattern advantages
# Make code more universal, separate code create object from code uses them
# reducing complexity
class Connector(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, is_secure):
        self.is_secure = is_secure
        self.port = self.port_factory_method()
        self.protocol = self.protocol_factory_method()

    def read(self, host, path):
        url = self.protocol + '://' + host + ':' + str(self.port) + path
        print('Connecting to....', url)
        return urllib2.urlopen(url, timeout=2).read()

    @abc.abstractmethod
    def protocol_factory_method(self):
        pass

    @abc.abstractmethod
    def port_factory_method(self):
        pass


class HTTPConnector(Connector):
    def protocol_factory_method(self):
        if self.is_secure:
            return 'https'
        return 'http'

    def port_factory_method(self):
        if self.is_secure:
            return HTTPSecurePort()
        return HTTPPort()

    def parse(self, content):
        filenames = []
        soup = BS(content, parser='html.parser')
        links = soup.table.findAll('a')
        for link in links:
            filenames.append(link['href'])
        return '\n'.join(filenames)


class Port(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __str__(self):
        pass


class HTTPPort(Port):
    def __str__(self):
        return '80'


class HTTPSecurePort(Port):
    def __str__(self):
        return '443'

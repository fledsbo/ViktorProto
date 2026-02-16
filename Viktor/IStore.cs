namespace Viktor
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    internal interface IStore
    {
        void UpsertItem(int key, byte[] data);

        bool ReadItem(int key, out byte[] data);

        IEnumerable<int> ReadAllItems();

        IEnumerable<int> FindItems(string query);
    }
}
